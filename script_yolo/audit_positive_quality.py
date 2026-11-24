#!/usr/bin/env python3
"""Audit likely low-quality positive samples without changing the dataset.

The report is intentionally a triage aid, not an automatic delete list. It
combines image-domain signals with YOLO boxes and manifest provenance so that
the highest-risk samples can be reviewed first.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageFilter, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    default_dir = project / "dataset/landslide_yolo_screened_2026-08-21"
    parser = argparse.ArgumentParser(description="筛查疑似低质量滑坡正样本")
    parser.add_argument("--data-dir", type=Path, default=default_dir)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--portrait-ratio", type=float, default=1.15)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=360)
    parser.add_argument("--blur-score", type=float, default=150.0)
    parser.add_argument("--tiny-box-ratio", type=float, default=0.003)
    return parser.parse_args()


def read_label(path: Path) -> tuple[int, float, bool]:
    boxes = 0
    min_area = 1.0
    clipped = False
    if not path.is_file():
        return 0, 0.0, False
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            _, cx, cy, width, height = map(float, fields)
        except ValueError:
            continue
        boxes += 1
        area = width * height
        min_area = min(min_area, area)
        clipped |= cx - width / 2 < 0.01 or cx + width / 2 > 0.99
        clipped |= cy - height / 2 < 0.01 or cy + height / 2 > 0.99
    return boxes, (0.0 if boxes == 0 else min_area), clipped


def image_scores(image: Image.Image) -> tuple[float, float, float]:
    gray = image.convert("L")
    brightness = float(ImageStat.Stat(gray).mean[0])
    extrema = gray.getextrema()
    dark_fraction = sum(1 for value in gray.resize((64, 64)).getdata() if value < 12) / 4096
    light_fraction = sum(1 for value in gray.resize((64, 64)).getdata() if value > 243) / 4096
    edges = gray.filter(ImageFilter.FIND_EDGES)
    blur_score = float(ImageStat.Stat(edges).var[0])
    exposure_fraction = max(dark_fraction, light_fraction)
    return brightness, blur_score, exposure_fraction


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest = data_dir / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"找不到 manifest.csv: {manifest}")
    output = (args.output or data_dir / "positive_quality_audit.csv").resolve()
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    for row in source_rows:
        if row.get("status") != "kept" or row.get("positive") != "1":
            continue
        image_path = (data_dir / row["output_path"]).resolve()
        label_path = data_dir / "labels" / row["split"] / f"{image_path.stem}.txt"
        reasons: list[str] = []
        width = height = 0
        brightness = blur = exposure = 0.0
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                brightness, blur, exposure = image_scores(image)
        except (OSError, ValueError) as exc:
            reasons.append(f"unreadable:{type(exc).__name__}")
        boxes, min_box_area, clipped = read_label(label_path)
        if width and height:
            if height / width >= args.portrait_ratio:
                reasons.append("portrait_domain_mismatch")
            if width < args.min_width or height < args.min_height:
                reasons.append("low_resolution")
            if blur < args.blur_score:
                reasons.append("possibly_blurred")
            if exposure >= 0.35:
                reasons.append("extreme_exposure")
        if boxes == 0:
            reasons.append("no_valid_box")
        elif min_box_area < args.tiny_box_ratio:
            reasons.append("tiny_target")
        if clipped:
            reasons.append("box_near_edge")
        rows.append({
            "path": str(image_path), "source": row.get("source", ""),
            "group": row.get("group", ""), "split": row.get("split", ""),
            "width": width, "height": height,
            "aspect_ratio": round(height / width, 4) if width else 0,
            "brightness": round(brightness, 2), "blur_score": round(blur, 2),
            "exposure_fraction": round(exposure, 4), "boxes": boxes,
            "min_box_area": round(min_box_area, 6), "reasons": ";".join(reasons),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["path", "reasons"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    flagged = [row for row in rows if row["reasons"]]
    print(f"positive={len(rows)} flagged={len(flagged)} output={output}")
    print("reasons=" + repr(Counter(reason for row in flagged for reason in row["reasons"].split(";") if reason)))


if __name__ == "__main__":
    main()
