#!/usr/bin/env python3
"""Mine false-positive train negatives and oversample them without copying data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="用当前模型扫描 train 负样本，并在训练清单中重复高置信度误报难例。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project / "dataset/landslide_yolo_gpt_clean_2026-08-26",
    )
    parser.add_argument("--model", required=True, help="用于难例挖掘的 .pt 模型。")
    parser.add_argument("--conf", type=float, default=0.20, help="误报入选最低置信度。")
    parser.add_argument("--hard-repeat", type=int, default=4, help="难负样本在最终清单中的总出现次数。")
    parser.add_argument("--max-hard", type=int, default=200, help="最多保留的困难负样本数，0 表示不限制。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--scan-chunk",
        type=int,
        default=64,
        help="每次交给 Ultralytics 的文件路径数，限制扫描内存峰值。",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--base-list", type=Path)
    parser.add_argument("--output-list", type=Path)
    parser.add_argument("--output-yaml", type=Path)
    parser.add_argument("--report-csv", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("conf 必须在 (0, 1) 范围内。")
    if args.hard_repeat < 1:
        parser.error("hard-repeat 必须至少为 1。")
    if args.max_hard < 0:
        parser.error("max-hard 不能小于 0。")
    if args.batch < 1 or args.scan_chunk < 1:
        parser.error("batch 和 scan-chunk 必须至少为 1。")
    args.data_dir = args.data_dir.resolve()
    args.base_list = (args.base_list or args.data_dir / "domain_balanced_train.txt").resolve()
    args.output_list = (args.output_list or args.data_dir / "hard_negative_train.txt").resolve()
    args.output_yaml = (args.output_yaml or args.data_dir / "data_hard_negative.yaml").resolve()
    args.report_csv = (args.report_csv or args.data_dir / "hard_negative_mining.csv").resolve()
    args.summary = (args.summary or args.data_dir / "hard_negative_mining_summary.json").resolve()
    return args


def load_train_negatives(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "kept"
            and row["split"] == "train"
            and row["positive"] == "0"
        ]
    for row in rows:
        image = (data_dir / row["output_path"]).resolve()
        label = data_dir / "labels/train" / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(image if not image.is_file() else label)
        if label.read_text(encoding="utf-8").strip():
            raise ValueError(f"manifest 标为负样本但标签非空: {label}")
        row["image_path"] = str(image)
    return rows


def rank_hard_negatives(
    rows: list[dict[str, str]], scores: dict[str, float], threshold: float, limit: int
) -> list[dict[str, str | float]]:
    ranked = [
        {**row, "max_confidence": scores.get(row["image_path"], 0.0)}
        for row in rows
        if scores.get(row["image_path"], 0.0) >= threshold
    ]
    ranked.sort(key=lambda row: (-float(row["max_confidence"]), row["image_path"]))
    return ranked if limit == 0 else ranked[:limit]


def main() -> None:
    args = parse_args()
    negatives = load_train_negatives(args.data_dir)
    paths = [row["image_path"] for row in negatives]
    model = YOLO(args.model)
    scores: dict[str, float] = {}
    for start in range(0, len(paths), args.scan_chunk):
        chunk = paths[start : start + args.scan_chunk]
        results = list(
            model.predict(
                source=chunk,
                conf=args.conf,
                iou=0.6,
                imgsz=args.imgsz,
                batch=min(args.batch, len(chunk)),
                device=args.device,
                stream=True,
                verbose=False,
            )
        )
        if len(results) != len(chunk):
            raise RuntimeError(
                f"分块推理结果数量不匹配: 输入 {len(chunk)}，输出 {len(results)}"
            )
        for image_path, result in zip(chunk, results):
            confidence = (
                float(result.boxes.conf.max().detach().cpu())
                if result.boxes is not None and len(result.boxes)
                else 0.0
            )
            scores[image_path] = confidence
    if len(scores) != len(paths):
        missing = sorted(set(paths) - set(scores))
        raise RuntimeError(f"推理结果数量不完整，缺少 {len(missing)} 张: {missing[:5]}")

    selected = rank_hard_negatives(negatives, scores, args.conf, args.max_hard)
    base_paths = [
        line.strip()
        for line in args.base_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_counts = {path: base_paths.count(path) for path in {row["image_path"] for row in selected}}
    extra_paths = []
    for row in selected:
        image_path = str(row["image_path"])
        extra_count = max(0, args.hard_repeat - base_counts.get(image_path, 0))
        extra_paths.extend([image_path] * extra_count)
        row["base_occurrences"] = base_counts.get(image_path, 0)
        row["extra_occurrences"] = extra_count
        row["final_occurrences"] = base_counts.get(image_path, 0) + extra_count

    final_paths = base_paths + extra_paths
    random.Random(args.seed).shuffle(final_paths)
    args.output_list.write_text("\n".join(final_paths) + "\n", encoding="utf-8")

    data = yaml.safe_load((args.data_dir / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(args.data_dir)
    data["train"] = str(args.output_list)
    args.output_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report_fields = [
        "rank",
        "max_confidence",
        "source",
        "group",
        "image_path",
        "original_path",
        "base_occurrences",
        "extra_occurrences",
        "final_occurrences",
    ]
    temporary_report = args.report_csv.with_name(f".{args.report_csv.name}.tmp")
    with temporary_report.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            writer.writerow(
                {field: rank if field == "rank" else row[field] for field in report_fields}
            )
    temporary_report.replace(args.report_csv)

    summary = {
        "data_dir": str(args.data_dir),
        "model": str(Path(args.model).resolve()),
        "threshold": args.conf,
        "hard_repeat": args.hard_repeat,
        "max_hard": args.max_hard,
        "scanned_train_negatives": len(negatives),
        "selected_hard_negatives": len(selected),
        "base_train_occurrences": len(base_paths),
        "extra_hard_negative_occurrences": len(extra_paths),
        "final_train_occurrences": len(final_paths),
        "selected_by_source": dict(
            sorted(
                {
                    source: sum(row["source"] == source for row in selected)
                    for source in {row["source"] for row in selected}
                }.items()
            )
        ),
        "top_hard_negatives": [
            {
                "max_confidence": row["max_confidence"],
                "source": row["source"],
                "image_path": row["image_path"],
            }
            for row in selected[:20]
        ],
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
