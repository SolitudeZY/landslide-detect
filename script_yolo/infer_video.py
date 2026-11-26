#!/usr/bin/env python3
"""Run YOLO detection with temporal alert hysteresis on a video or stream."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def alert_ready(votes: deque[bool], window: int, min_hits: int) -> bool:
    """Compatibility helper for the former inference tests."""
    return len(votes) == window and sum(votes) >= min_hits


@dataclass
class TemporalAlert:
    window: int = 10
    activate_hits: int = 6
    clear_misses: int = 20
    votes: deque[bool] = field(init=False)
    active: bool = False
    miss_streak: int = 0

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window 必须至少为 1")
        if not 1 <= self.activate_hits <= self.window:
            raise ValueError("activate_hits 必须在 1 到 window 之间")
        if self.clear_misses < 1:
            raise ValueError("clear_misses 必须至少为 1")
        self.votes = deque(maxlen=self.window)

    def update(self, detected: bool) -> tuple[bool, str | None]:
        self.votes.append(detected)
        if self.active:
            self.miss_streak = 0 if detected else self.miss_streak + 1
            if self.miss_streak >= self.clear_misses:
                self.active = False
                self.miss_streak = 0
                return False, "cleared"
            return True, None
        if len(self.votes) < self.window:
            return self.active, None
        hits = sum(self.votes)
        if hits >= self.activate_hits:
            self.active = True
            self.miss_streak = 0 if detected else 1
            return True, "activated"
        return self.active, None

    @property
    def hits(self) -> int:
        return sum(self.votes)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="对视频、RTSP 或摄像头执行 YOLO 滑坡检测和时序报警。"
    )
    parser.add_argument("--model", required=True, help="best.pt 或导出的 ONNX。")
    parser.add_argument("--source", required=True, help="视频路径、RTSP URL 或摄像头编号。")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--events", type=Path, help="默认与输出视频同名的 .jsonl。")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--alert-window", type=int, default=10)
    parser.add_argument("--alert-min-hits", type=int, default=6)
    parser.add_argument(
        "--alert-clear-misses",
        type=int,
        default=20,
        help="报警后连续多少帧无检测才解除；应按实际推理 FPS 标定。",
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="仅输出 JSONL，不保存标注视频。")
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示不限制，便于冒烟测试。")
    parser.add_argument("--runs-dir", type=Path, default=script_dir / "runs/infer")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("conf 必须在 (0, 1) 范围内。")
    if args.alert_window < 1 or not 1 <= args.alert_min_hits <= args.alert_window:
        parser.error("alert-min-hits 必须在 1 到 alert-window 之间。")
    if args.alert_clear_misses < 1:
        parser.error("alert-clear-misses 必须至少为 1。")
    return args


def default_output(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output.resolve()
    source_path = Path(args.source)
    stem = source_path.stem if source_path.suffix else f"stream_{int(time.time())}"
    return (args.runs_dir / f"{stem}_annotated.mp4").resolve()


def draw_detections(frame, detections: list[dict], alert: bool, votes: int, window: int):
    annotated = frame.copy()
    line_width = max(2, round(min(frame.shape[:2]) / 300))
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection["xyxy"])
        color = (0, 0, 255) if alert else (0, 190, 255)
        label = f"landslide {detection['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
        text_y = y1 - 8 if y1 >= 28 else y1 + 22
        cv2.putText(annotated, label, (max(2, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    status = "LANDSLIDE ALERT" if alert else "MONITORING"
    color = (0, 0, 255) if alert else (0, 180, 0)
    status_text = f"{status}  detections={votes}/{window}"
    (text_width, text_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(annotated, (10, 10), (25 + text_width, 24 + text_height), (0, 0, 0), -1)
    cv2.putText(annotated, status_text, (18, 18 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return annotated


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    output_path = default_output(args)
    events_path = (args.events or output_path.with_suffix(".jsonl")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频源: {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 25.0
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"无法创建输出视频: {output_path}")

    model = YOLO(args.model)
    temporal = TemporalAlert(args.alert_window, args.alert_min_hits, args.alert_clear_misses)
    frame_index = detection_frames = alert_frames = activations = clears = 0
    try:
        with events_path.open("w", encoding="utf-8") as event_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.detach().cpu().tolist()
                    confidences = result.boxes.conf.detach().cpu().tolist()
                    classes = result.boxes.cls.detach().cpu().tolist()
                    for box, confidence, class_id in zip(xyxy, confidences, classes):
                        detections.append({
                            "class_id": int(class_id),
                            "class_name": result.names[int(class_id)],
                            "confidence": round(float(confidence), 6),
                            "xyxy": [round(float(value), 2) for value in box],
                        })
                detection_frames += bool(detections)
                alert, transition = temporal.update(bool(detections))
                activations += transition == "activated"
                clears += transition == "cleared"
                alert_frames += alert
                event = {
                    "frame": frame_index,
                    "timestamp_ms": round(float(capture.get(cv2.CAP_PROP_POS_MSEC)), 2),
                    "detections": detections,
                    "detection_count": len(detections),
                    "temporal_hits": temporal.hits,
                    "temporal_window_filled": len(temporal.votes) == temporal.window,
                    "alert": alert,
                    "alert_transition": transition,
                }
                event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                annotated = draw_detections(frame, detections, alert, temporal.hits, temporal.window)
                if writer is not None:
                    writer.write(annotated)
                if args.view:
                    cv2.imshow("landslide-yolo", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
                if args.max_frames and frame_index >= args.max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.view:
            cv2.destroyAllWindows()

    print(json.dumps({
        "frames": frame_index,
        "detection_frames": detection_frames,
        "alert_frames": alert_frames,
        "alert_activations": activations,
        "alert_clears": clears,
        "alert_rule": f"activate>={args.alert_min_hits}/{args.alert_window}, clear_after_misses={args.alert_clear_misses}",
        "temporal_mode": "detector_hysteresis",
        "video": None if args.no_video else str(output_path),
        "events": str(events_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
