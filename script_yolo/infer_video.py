#!/usr/bin/env python3
"""Run YOLO on a video/stream and emit an annotated video plus JSONL events."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from motion import MotionAnalyzer, MotionConfig


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="对视频、RTSP 或摄像头执行滑坡检测。")
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
    parser.add_argument("--disable-motion-gate", action="store_true")
    parser.add_argument("--motion-width", type=int, default=640)
    parser.add_argument("--motion-frame-gap", type=int, default=5)
    parser.add_argument("--diff-pixel-threshold", type=int, default=18)
    parser.add_argument("--diff-ratio-threshold", type=float, default=0.003)
    parser.add_argument("--flow-threshold", type=float, default=0.20)
    parser.add_argument("--min-flow-points", type=int, default=4)
    parser.add_argument(
        "--max-camera-motion",
        type=float,
        default=3.0,
        help="5 帧跨度内允许的全局位移像素；负数表示关闭相机稳定性门。",
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="仅输出 JSONL，不保存标注视频。")
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示不限制，便于冒烟测试。")
    parser.add_argument("--runs-dir", type=Path, default=script_dir / "runs/infer")
    args = parser.parse_args()
    if args.alert_window < 1 or not 1 <= args.alert_min_hits <= args.alert_window:
        parser.error("alert-min-hits 必须在 1 到 alert-window 之间。")
    if args.motion_width < 64 or args.motion_frame_gap < 1:
        parser.error("motion-width 至少为 64，motion-frame-gap 至少为 1。")
    if not 0 <= args.diff_pixel_threshold <= 255:
        parser.error("diff-pixel-threshold 必须在 0 到 255 之间。")
    if not 0 <= args.diff_ratio_threshold <= 1 or args.flow_threshold < 0:
        parser.error("帧差比例必须在 0 到 1 之间，光流阈值不能为负。")
    if args.min_flow_points < 1:
        parser.error("min-flow-points 至少为 1。")
    return args


def default_output(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output.resolve()
    source_path = Path(args.source)
    stem = source_path.stem if source_path.suffix else f"stream_{int(time.time())}"
    return (args.runs_dir / f"{stem}_annotated.mp4").resolve()


def alert_ready(votes: deque[bool], window: int, min_hits: int) -> bool:
    return len(votes) == window and sum(votes) >= min_hits


def draw_detections(
    frame,
    detections: list[dict],
    alert: bool,
    votes: int,
    window: int,
):
    annotated = frame.copy()
    line_width = max(2, round(min(frame.shape[:2]) / 300))
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection["xyxy"])
        confirmed = detection["motion_confirmed"]
        color = (0, 190, 0) if confirmed else (0, 190, 255)
        motion = detection.get("motion")
        state = "MOTION" if confirmed else "STATIC"
        if motion and motion["available"]:
            detail = (
                f"D={motion['diff_ratio']:.3f} F={motion['flow_magnitude']:.2f} "
                f"C={motion['camera_motion']:.1f}"
            )
        elif motion:
            detail = "motion warmup"
        else:
            detail = "motion gate off"
        label = f"landslide {detection['confidence']:.2f} {state} {detail}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
        text_y = y1 - 8 if y1 >= 28 else y1 + 22
        cv2.putText(
            annotated,
            label,
            (max(2, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    status = "LANDSLIDE ALERT" if alert else "MONITORING"
    color = (0, 0, 255) if alert else (0, 180, 0)
    status_text = f"{status}  motion votes={votes}/{window}"
    (text_width, text_height), _ = cv2.getTextSize(
        status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(annotated, (10, 10), (25 + text_width, 24 + text_height), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        status_text,
        (18, 18 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
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
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"无法创建输出视频: {output_path}")

    model = YOLO(args.model)
    motion_analyzer = MotionAnalyzer(
        MotionConfig(
            analysis_width=args.motion_width,
            frame_gap=args.motion_frame_gap,
            diff_pixel_threshold=args.diff_pixel_threshold,
            diff_ratio_threshold=args.diff_ratio_threshold,
            flow_threshold=args.flow_threshold,
            min_flow_points=args.min_flow_points,
            max_camera_motion=args.max_camera_motion,
        )
    )
    votes: deque[bool] = deque(maxlen=args.alert_window)
    frame_index = 0
    raw_detection_frames = 0
    motion_detection_frames = 0
    alert_frames = 0
    try:
        with events_path.open("w", encoding="utf-8") as event_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                detections = []
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.detach().cpu().tolist()
                    confidences = result.boxes.conf.detach().cpu().tolist()
                    classes = result.boxes.cls.detach().cpu().tolist()
                    metrics = (
                        [None] * len(xyxy)
                        if args.disable_motion_gate
                        else motion_analyzer.analyze(frame, xyxy)
                    )
                    for box, confidence, class_id, motion in zip(
                        xyxy, confidences, classes, metrics
                    ):
                        detections.append(
                            {
                                "class_id": int(class_id),
                                "class_name": result.names[int(class_id)],
                                "confidence": round(float(confidence), 6),
                                "xyxy": [round(float(value), 2) for value in box],
                                "motion_confirmed": (
                                    True if motion is None else motion.confirmed
                                ),
                                "motion": None if motion is None else motion.to_dict(),
                            }
                        )
                elif not args.disable_motion_gate:
                    motion_analyzer.analyze(frame, [])
                motion_detections = [
                    detection
                    for detection in detections
                    if detection["motion_confirmed"]
                ]
                raw_detection_frames += bool(detections)
                motion_detection_frames += bool(motion_detections)
                votes.append(bool(motion_detections))
                alert = alert_ready(votes, args.alert_window, args.alert_min_hits)
                alert_frames += alert
                timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                event = {
                    "frame": frame_index,
                    "timestamp_ms": round(float(timestamp_ms), 2),
                    "detections": detections,
                    "raw_detection_count": len(detections),
                    "motion_detection_count": len(motion_detections),
                    "temporal_votes": sum(votes),
                    "alert": alert,
                }
                event_file.write(json.dumps(event, ensure_ascii=False) + "\n")

                annotated = draw_detections(
                    frame, detections, alert, sum(votes), args.alert_window
                )
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

    print(
        json.dumps(
            {
                "frames": frame_index,
                "raw_detection_frames": raw_detection_frames,
                "motion_detection_frames": motion_detection_frames,
                "alert_frames": alert_frames,
                "alert_rule": f"{args.alert_min_hits}/{args.alert_window}",
                "motion_gate": not args.disable_motion_gate,
                "video": None if args.no_video else str(output_path),
                "events": str(events_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
