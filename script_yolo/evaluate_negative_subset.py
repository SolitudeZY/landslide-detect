#!/usr/bin/env python3
"""Measure image-level false positives on a manifest-defined negative subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


DEFAULT_SOURCES = ("negative_20260821", "reviewed_negative_20260821")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="评估指定负样本子集的图片级误报率。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "dataset/landslide_yolo_screened_2026-08-21",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--threshold", action="append", type=float, dest="thresholds")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.sources = tuple(args.sources or DEFAULT_SOURCES)
    args.thresholds = sorted(set(args.thresholds or (0.25, 0.35, 0.50)))
    if not args.thresholds or not all(0 < value <= 1 for value in args.thresholds):
        parser.error("threshold 必须在 (0, 1] 范围内。")
    return args


def select_images(data_dir: Path, split: str, sources: tuple[str, ...]) -> list[Path]:
    manifest = data_dir / "manifest.csv"
    selected: list[Path] = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["status"] == "kept"
                and (split == "all" or row["split"] == split)
                and row["source"] in sources
                and row["positive"] == "0"
            ):
                image_path = data_dir / row["output_path"]
                label_path = (
                    data_dir
                    / "labels"
                    / row["split"]
                    / f"{image_path.stem}.txt"
                )
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if not label_path.is_file() or label_path.read_text().strip():
                    raise ValueError(f"所选样本不是空标签负样本: {image_path}")
                selected.append(image_path)
    if not selected:
        raise ValueError(f"没有匹配的负样本: split={split}, sources={sources}")
    return sorted(selected)


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_dir = args.data_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    images = select_images(data_dir, args.split, args.sources)
    minimum_confidence = min(args.thresholds)
    predictions: list[dict[str, object]] = []
    results = YOLO(str(model_path)).predict(
        source=[str(path) for path in images],
        conf=minimum_confidence,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        save=False,
    )
    for image_path, result in zip(images, results, strict=True):
        confidences = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
        predictions.append(
            {
                "image": str(image_path),
                "confidences": sorted((float(value) for value in confidences), reverse=True),
            }
        )

    metrics: dict[str, object] = {}
    for threshold in args.thresholds:
        false_positive_images = [
            item
            for item in predictions
            if any(value >= threshold for value in item["confidences"])
        ]
        detections = sum(
            value >= threshold
            for item in predictions
            for value in item["confidences"]
        )
        metrics[f"{threshold:.2f}"] = {
            "images": len(images),
            "false_positive_images": len(false_positive_images),
            "image_false_positive_rate": len(false_positive_images) / len(images),
            "false_positive_detections": detections,
            "false_positive_files": [item["image"] for item in false_positive_images],
        }

    report = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "split": args.split,
        "sources": list(args.sources),
        "imgsz": args.imgsz,
        "metrics": metrics,
        "maximum_confidence": max(
            (value for item in predictions for value in item["confidences"]),
            default=0.0,
        ),
        "predictions": predictions,
    }
    output = args.output or (
        Path(__file__).resolve().parent
        / "runs/eval"
        / f"{model_path.stem}_{args.split}_new_negatives.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
