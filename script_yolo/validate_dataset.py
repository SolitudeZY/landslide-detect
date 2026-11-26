#!/usr/bin/env python3
"""Validate image/label pairing, YOLO coordinates, and split leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    default_data = (
        Path(__file__).resolve().parent.parent
        / "dataset/landslide_yolo_screened_2026-08-21/data.yaml"
    )
    parser = argparse.ArgumentParser(description="验证融合后的 YOLO 数据集。")
    parser.add_argument("--data", type=Path, default=default_data)
    return parser.parse_args()


def resolve_root(data_path: Path, config: dict) -> Path:
    data_path = data_path.resolve()
    configured_root = config.get("path")
    root = Path(configured_root) if configured_root else data_path.parent
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = resolve_root(args.data, config)
    errors: list[str] = []
    hashes: dict[str, tuple[str, Path]] = {}
    counts: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        label_dir = root / "labels" / split
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem for path in images}
        orphan_labels = set(labels) - image_stems
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} 个标签没有对应图片")

        positive = 0
        negative = 0
        boxes = 0
        for image_path in images:
            label_path = labels.get(image_path.stem)
            if label_path is None:
                errors.append(f"{split}: 图片缺少标签 {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except OSError as exc:
                errors.append(f"{split}: 图片损坏 {image_path}: {exc}")
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            previous = hashes.get(digest)
            if previous is not None:
                errors.append(
                    f"SHA-256 重复且可能跨划分: {previous[0]}:{previous[1]} <-> {split}:{image_path}"
                )
            else:
                hashes[digest] = (split, image_path)

            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if lines:
                positive += 1
            else:
                negative += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label_path}:{line_number}: 应有 5 列")
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: 非法数值")
                    continue
                if class_id != 0:
                    errors.append(f"{label_path}:{line_number}: class id 不是 0")
                if any(value <= 0.0 or value > 1.0 for value in values[2:]):
                    errors.append(f"{label_path}:{line_number}: 宽高越界 {values[2:]}")
                if any(value < 0.0 or value > 1.0 for value in values[:2]):
                    errors.append(f"{label_path}:{line_number}: 中心点越界 {values[:2]}")
                center_x, center_y, width, height = values
                if (
                    center_x - width / 2 < -1e-6
                    or center_x + width / 2 > 1 + 1e-6
                    or center_y - height / 2 < -1e-6
                    or center_y + height / 2 > 1 + 1e-6
                ):
                    errors.append(f"{label_path}:{line_number}: 框超出图像边界")
                boxes += 1
        counts[split] = {
            "images": len(images),
            "positive": positive,
            "negative": negative,
            "boxes": boxes,
        }

    manifest_path = root / "manifest.csv"
    if manifest_path.is_file():
        group_splits: dict[str, set[str]] = {}
        for row in read_csv_rows(manifest_path):
            if row["status"] != "kept" or not row["group"]:
                continue
            group_splits.setdefault(row["group"], set()).add(row["split"])
        leaked_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
        if leaked_groups:
            errors.append(f"group_leakage: {len(leaked_groups)} 个来源组跨越数据划分")

    result = {
        "data": str(args.data.resolve()),
        "counts": counts,
        "errors": len(errors),
        "error_types": dict(Counter(error.split(":", 1)[0] for error in errors)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
