#!/usr/bin/env python3
"""Measure annotated-box recall on a manifest-defined positive subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="评估指定正样本子集的标注框召回率。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "dataset/landslide_yolo_screened_2026-08-21",
    )
    parser.add_argument("--split", default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--source", default="new_positive_20260821")
    parser.add_argument("--threshold", action="append", type=float, dest="thresholds")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.thresholds = sorted(set(args.thresholds or (0.25, 0.35, 0.50)))
    if not all(0 < value <= 1 for value in args.thresholds):
        parser.error("threshold 必须在 (0, 1] 范围内。")
    if not 0 < args.iou <= 1:
        parser.error("iou 必须在 (0, 1] 范围内。")
    return args


def load_samples(
    data_dir: Path, split: str, source: str
) -> list[tuple[Path, Path, str]]:
    samples: list[tuple[Path, Path, str]] = []
    with (data_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["status"] == "kept"
                and row["positive"] == "1"
                and row["source"] == source
                and (split == "all" or row["split"] == split)
            ):
                image_path = data_dir / row["output_path"]
                label_path = data_dir / "labels" / row["split"] / f"{image_path.stem}.txt"
                if not image_path.is_file() or not label_path.is_file():
                    raise FileNotFoundError(image_path)
                if not label_path.read_text().strip():
                    raise ValueError(f"所选正样本没有标注框: {image_path}")
                samples.append((image_path, label_path, row["split"]))
    if not samples:
        raise ValueError(f"没有匹配的正样本: split={split}, source={source}")
    return sorted(samples)


def read_ground_truth(label_path: Path, width: int, height: int) -> torch.Tensor:
    boxes = []
    for line in label_path.read_text().splitlines():
        _, cx, cy, box_width, box_height = (float(value) for value in line.split())
        boxes.append(
            [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ]
        )
    return torch.tensor(boxes, dtype=torch.float32)


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection_min = torch.maximum(left[:, None, :2], right[None, :, :2])
    intersection_max = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (intersection_max - intersection_min).clamp(min=0).prod(2)
    left_area = (left[:, 2:] - left[:, :2]).prod(1)[:, None]
    right_area = (right[:, 2:] - right[:, :2]).prod(1)[None, :]
    return intersection / (left_area + right_area - intersection).clamp(min=1e-9)


def matched_ground_truth(
    ground_truth: torch.Tensor, predicted: torch.Tensor, iou_threshold: float
) -> int:
    if ground_truth.numel() == 0 or predicted.numel() == 0:
        return 0
    ious = box_iou(ground_truth, predicted)
    matched = 0
    while ious.numel() and float(ious.max()) >= iou_threshold:
        flat_index = int(ious.argmax())
        gt_index = flat_index // ious.shape[1]
        prediction_index = flat_index % ious.shape[1]
        matched += 1
        ious[gt_index, :] = -1
        ious[:, prediction_index] = -1
    return matched


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_dir = args.data_dir.resolve()
    samples = load_samples(data_dir, args.split, args.source)
    results = YOLO(str(model_path)).predict(
        source=[str(image_path) for image_path, _, _ in samples],
        conf=min(args.thresholds),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        save=False,
    )
    predictions: list[dict[str, object]] = []
    for (image_path, label_path, sample_split), result in zip(samples, results, strict=True):
        height, width = result.orig_shape
        ground_truth = read_ground_truth(label_path, width, height)
        predicted_boxes = (
            torch.empty((0, 4))
            if result.boxes is None
            else result.boxes.xyxy.cpu()
        )
        confidences = (
            torch.empty((0,))
            if result.boxes is None
            else result.boxes.conf.cpu()
        )
        threshold_results = {}
        for threshold in args.thresholds:
            selected = predicted_boxes[confidences >= threshold]
            matched = matched_ground_truth(ground_truth, selected, args.iou)
            threshold_results[f"{threshold:.2f}"] = {
                "ground_truth_boxes": len(ground_truth),
                "predicted_boxes": len(selected),
                "matched_ground_truth": matched,
            }
        predictions.append(
            {
                "image": str(image_path),
                "split": sample_split,
                "thresholds": threshold_results,
            }
        )

    metrics = {}
    for threshold in args.thresholds:
        key = f"{threshold:.2f}"
        ground_truth_boxes = sum(item["thresholds"][key]["ground_truth_boxes"] for item in predictions)
        matched = sum(item["thresholds"][key]["matched_ground_truth"] for item in predictions)
        complete_misses = [
            item["image"]
            for item in predictions
            if item["thresholds"][key]["matched_ground_truth"] == 0
        ]
        partial_or_complete_misses = [
            item["image"]
            for item in predictions
            if item["thresholds"][key]["matched_ground_truth"]
            < item["thresholds"][key]["ground_truth_boxes"]
        ]
        metrics[key] = {
            "images": len(predictions),
            "ground_truth_boxes": ground_truth_boxes,
            "matched_ground_truth": matched,
            "box_recall": matched / ground_truth_boxes,
            "complete_miss_images": len(complete_misses),
            "images_with_any_missed_box": len(partial_or_complete_misses),
            "complete_miss_files": complete_misses,
            "any_missed_box_files": partial_or_complete_misses,
        }
    report = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "split": args.split,
        "source": args.source,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "metrics": metrics,
        "predictions": predictions,
    }
    output = args.output or (
        Path(__file__).resolve().parent
        / "runs/eval"
        / f"{model_path.stem}_{args.split}_{args.source}.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
