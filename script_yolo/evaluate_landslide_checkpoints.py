#!/usr/bin/env python3
"""Compare checkpoints on held-out negatives and the staged Seedance video."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from infer_video import TemporalAlert


THRESHOLDS = (0.25, 0.35, 0.50)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="按独立负样本与阶段视频比较模型。")
    parser.add_argument("--model", action="append", required=True, help="name=path，可重复传入。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project / "dataset/landslide_yolo_gpt_clean_2026-08-26",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=project / "script_yolo/test-video-ai/seedance-landslide-test.mp4",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    models = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"model 必须是 name=path: {value}")
        name, path = value.split("=", 1)
        resolved = Path(path).resolve()
        if not name or not resolved.is_file():
            raise FileNotFoundError(resolved)
        models.append((name, resolved))
    return models


def predict_scores(model: YOLO, paths: list[str], args: argparse.Namespace) -> dict[str, float]:
    scores = {}
    for start in range(0, len(paths), args.chunk):
        chunk = paths[start : start + args.chunk]
        results = list(
            model.predict(
                source=chunk,
                conf=0.05,
                iou=0.6,
                imgsz=args.imgsz,
                batch=min(args.batch, len(chunk)),
                device=args.device,
                stream=True,
                verbose=False,
            )
        )
        if len(results) != len(chunk):
            raise RuntimeError("静态图片推理结果数与输入不一致")
        for path, result in zip(chunk, results):
            scores[path] = (
                float(result.boxes.conf.max().detach().cpu())
                if result.boxes is not None and len(result.boxes)
                else 0.0
            )
    return scores


def video_scores(model: YOLO, video: Path, args: argparse.Namespace) -> list[float]:
    scores = []
    for result in model.predict(
        source=str(video),
        conf=0.05,
        iou=0.6,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    ):
        scores.append(
            float(result.boxes.conf.max().detach().cpu())
            if result.boxes is not None and len(result.boxes)
            else 0.0
        )
    return scores


def threshold_counts(scores: list[float]) -> dict[str, int]:
    return {f"conf_{threshold:.2f}": sum(score >= threshold for score in scores) for threshold in THRESHOLDS}


def temporal_summary(scores: list[float], threshold: float = 0.35) -> dict[str, object]:
    state = TemporalAlert(window=10, activate_hits=6, clear_misses=20)
    activations = []
    clears = []
    alert_frames = 0
    for frame, score in enumerate(scores):
        alert, transition = state.update(score >= threshold)
        alert_frames += alert
        if transition == "activated":
            activations.append(frame)
        elif transition == "cleared":
            clears.append(frame)
    return {"activations": activations, "clears": clears, "alert_frames": alert_frames}


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.video = args.video.resolve()
    models = parse_models(args.model)
    with (args.data_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    heldout_rows = [
        row
        for row in rows
        if row["status"] == "kept"
        and row["positive"] == "0"
        and row["split"] in {"val", "test"}
    ]
    heldout_paths = [str((args.data_dir / row["output_path"]).resolve()) for row in heldout_rows]

    hard_report = args.data_dir / "hard_negative_mining_round1.csv"
    with hard_report.open(encoding="utf-8-sig", newline="") as handle:
        hard_paths = [row["image_path"] for row in csv.DictReader(handle)]

    report = {
        "video": str(args.video),
        "heldout_negative_images": len(heldout_paths),
        "round1_hard_negative_images": len(hard_paths),
        "models": {},
    }
    for name, path in models:
        model = YOLO(str(path))
        heldout = predict_scores(model, heldout_paths, args)
        hard = predict_scores(model, hard_paths, args)
        staged = video_scores(model, args.video, args)
        if len(staged) != 193:
            raise RuntimeError(f"Seedance 视频预期193帧，实际 {len(staged)}")
        report["models"][name] = {
            "path": str(path),
            "heldout_negatives": {
                **threshold_counts(list(heldout.values())),
                "max_confidence": max(heldout.values(), default=0.0),
            },
            "round1_hard_negatives": {
                **threshold_counts(list(hard.values())),
                "max_confidence": max(hard.values(), default=0.0),
                "mean_confidence": sum(hard.values()) / len(hard) if hard else 0.0,
            },
            "seedance_video": {
                "normal_frames_0_72": threshold_counts(staged[0:73]),
                "transition_frames_73_77": threshold_counts(staged[73:78]),
                "active_frames_78_156": threshold_counts(staged[78:157]),
                "late_frames_157_192": threshold_counts(staged[157:193]),
                "normal_max_confidence": max(staged[0:73]),
                "active_max_confidence": max(staged[78:157]),
                "late_max_confidence": max(staged[157:193]),
                "temporal_conf_0.35": temporal_summary(staged),
            },
        }
        del model
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
