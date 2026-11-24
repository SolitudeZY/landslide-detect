#!/usr/bin/env python3
"""Create a split-safe YOLO dataset augmented with reviewed GPT landslides."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


SPLITS = ("train", "val", "test")
GPT_INPUTS = (
    ("gpt_v1", "landslide-gpt-v1/annotations/instances_Train.json", "landslide-gpt-v1/images/Train"),
    ("gpt_v2", "landslide-gpt-v2/annotations/instances_Validation.json", "landslide-gpt-v2/images/Validation"),
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="融合人工审核的 GPT 滑坡正样本")
    parser.add_argument(
        "--base-data-dir", type=Path,
        default=project / "dataset/landslide_yolo_screened_2026-08-21",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project / "dataset/landslide_yolo_gpt_clean_2026-08-26",
    )
    parser.add_argument(
        "--group-cap", type=int, default=0,
        help="同一底图最多保留的生成变体；0 表示不限制（默认）。",
    )
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def parent_name(filename: str) -> str:
    name = re.sub(r"^\d{6}_", "", filename)
    return re.sub(r"_(?:small|medium|large)_(?:left|right)(?=\.[^.]+$)", "", name)


def variant_rank(filename: str, source: str) -> tuple[int, int, str, str]:
    scale = next((item for item in ("medium", "large", "small") if f"_{item}_" in filename), "other")
    side = next((item for item in ("left", "right") if f"_{item}." in filename), "other")
    return ({"medium": 0, "large": 1, "small": 2, "other": 0}[scale], {"left": 0, "right": 1, "other": 0}[side], source, filename)


def load_gpt_samples(project: Path, base_original_splits: dict[str, str], group_cap: int) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    audit: list[dict] = []
    for source, annotation_rel, image_rel in GPT_INPUTS:
        annotation_path = project / "dataset" / annotation_rel
        image_dir = project / "dataset" / image_rel
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
        if set(categories.values()) != {"landslide"}:
            raise ValueError(f"类别不正确: {annotation_path}: {categories}")
        images = {int(item["id"]): item for item in coco.get("images", [])}
        annotations: dict[int, list[dict]] = defaultdict(list)
        for item in coco.get("annotations", []):
            if item.get("category_id") not in categories:
                raise ValueError(f"未知类别: {item}")
            annotations[int(item["image_id"])].append(item)
        for image_id, image in images.items():
            path = image_dir / image["file_name"]
            base = parent_name(image["file_name"])
            record = {
                "source": source,
                "path": str(path.resolve()),
                "filename": image["file_name"],
                "parent_name": base,
                "parent_split": base_original_splits.get(base, "train"),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "boxes": [item["bbox"] for item in annotations[image_id]],
                "status": "candidate",
                "note": "",
            }
            if not path.is_file():
                record["status"] = "excluded_missing"
                record["note"] = "图片文件不存在"
            elif not record["boxes"]:
                record["status"] = "excluded_ambiguous_unlabeled"
                record["note"] = "无标注生成图含疑似滑坡，不能作为纯负样本"
            else:
                for x, y, width, height in record["boxes"]:
                    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > record["width"] or y + height > record["height"]:
                        record["status"] = "excluded_invalid_box"
                        record["note"] = "COCO 框越界或无效"
                        break
            audit.append(record)
            if record["status"] == "candidate":
                candidates.append(record)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[item["parent_name"]].append(item)
    selected: list[dict] = []
    for parent, items in grouped.items():
        ordered = sorted(items, key=lambda item: variant_rank(item["filename"], item["source"]))
        keep_count = len(ordered) if group_cap == 0 else group_cap
        for item in ordered[:keep_count]:
            item["status"] = "selected"
            item["note"] = "人工标注有效；按底图划分分配"
            selected.append(item)
        for item in ordered[keep_count:]:
            item["status"] = "excluded_variant_cap"
            item["note"] = f"同一底图最多保留 {group_cap} 个生成变体"
    return selected, audit


def write_calibration(
    active_rows: list[dict[str, str]], staging: Path, final_output: Path
) -> None:
    rng = random.Random(20260826)
    real_rows = [row for row in active_rows if not row["source"].startswith("gpt_")]
    positives = [row for row in real_rows if row["positive"] == "1" and row["split"] == "train"]
    negatives = [row for row in real_rows if row["positive"] == "0" and row["split"] == "train"]
    rng.shuffle(positives); rng.shuffle(negatives)
    selected = positives[:250] + negatives[:250]
    rng.shuffle(selected)
    paths = [str((final_output / row["output_path"]).resolve()) for row in selected]
    (staging / "calibration.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (staging / "calibration_wsl.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


def real_test_paths(active_rows: list[dict[str, str]], output: Path) -> list[str]:
    return [
        str((output / row["output_path"]).resolve())
        for row in active_rows
        if row["status"] == "kept"
        and row["split"] == "test"
        and not row["source"].startswith("gpt_")
    ]


def main() -> None:
    args = parse_args()
    if args.group_cap < 0:
        raise SystemExit("group-cap 必须不小于 0")
    base = args.base_data_dir.resolve()
    output = args.output.resolve()
    project = Path(__file__).resolve().parent.parent
    if output.exists() and not args.overwrite:
        raise SystemExit(f"输出已存在；如确认覆盖请加 --overwrite: {output}")
    staging = output.with_name(f".{output.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    if output.exists():
        shutil.rmtree(output)
    for split in SPLITS:
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    (staging / "annotations").mkdir()

    manifest_rows = read_csv(base / "manifest.csv")
    active = [row for row in manifest_rows if row["status"] == "kept"]
    base_original_splits = {Path(row["original_path"]).name: row["split"] for row in active}
    selected, audit = load_gpt_samples(project, base_original_splits, args.group_cap)
    for row in active:
        image = base / row["output_path"]
        label = base / "labels" / row["split"] / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(image)
        transfer(image, staging / row["output_path"], args.copy_mode)
        transfer(label, staging / "labels" / row["split"] / label.name, args.copy_mode)

    coco = {}
    for split in SPLITS:
        coco[split] = json.loads((base / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
    existing_names = {Path(row["output_path"]).name for row in active}
    for item in selected:
        source_path = Path(item["path"])
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        output_name = f"{digest[:16]}_{item['source']}{source_path.suffix.lower()}"
        if output_name in existing_names:
            raise ValueError(f"输出名称冲突: {output_name}")
        existing_names.add(output_name)
        split = item["parent_split"]
        transfer(source_path, staging / "images" / split / output_name, args.copy_mode)
        lines = []
        for x, y, width, height in item["boxes"]:
            lines.append(f"0 {(x + width / 2) / item['width']:.8f} {(y + height / 2) / item['height']:.8f} {width / item['width']:.8f} {height / item['height']:.8f}")
        (staging / "labels" / split / f"{Path(output_name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        group = f"gpt:{item['parent_name']}"
        manifest_rows.append({
            "status": "kept", "source": item["source"], "positive": "1", "split": split,
            "group": group, "original_path": str(source_path), "output_path": f"images/{split}/{output_name}",
            "width": str(item["width"]), "height": str(item["height"]), "box_count": str(len(item["boxes"])),
            "sha256": digest, "duplicate_of": "", "note": "GPT generated; manually annotated; split follows parent image",
        })
        split_coco = coco[split]
        image_id = max((int(x["id"]) for x in split_coco["images"]), default=0) + 1
        split_coco["images"].append({"id": image_id, "file_name": f"../images/{split}/{output_name}", "width": item["width"], "height": item["height"], "source": item["source"]})
        next_annotation_id = max((int(x["id"]) for x in split_coco["annotations"]), default=0) + 1
        for index, box in enumerate(item["boxes"]):
            split_coco["annotations"].append({"id": next_annotation_id + index, "image_id": image_id, "category_id": 1, "bbox": box, "area": box[2] * box[3], "iscrowd": 0, "segmentation": []})

    fields = list(manifest_rows[0])
    with (staging / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest_rows)
    for split in SPLITS:
        (staging / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco[split], ensure_ascii=False, indent=2), encoding="utf-8")
    data = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "landslide"}}
    (staging / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    active_rows = [row for row in manifest_rows if row["status"] == "kept"]
    write_calibration(active_rows, staging, output)
    real_test = real_test_paths(active_rows, output)
    (staging / "real_test.txt").write_text("\n".join(real_test) + "\n", encoding="utf-8")
    real_test_data = {"path": str(output), "train": "images/train", "val": "images/val", "test": str((output / "real_test.txt").resolve()), "names": {0: "landslide"}}
    (staging / "data_real_test.yaml").write_text(yaml.safe_dump(real_test_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    audit_rows = [{key: value for key, value in item.items() if key != "boxes"} | {"box_count": len(item["boxes"])} for item in audit]
    (staging / "gpt_quality_audit.json").write_text(json.dumps({"group_cap": args.group_cap, "records": audit_rows, "status_counts": dict(Counter(item["status"] for item in audit))}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "base_data_dir": str(base), "gpt_selection": dict(Counter(item["source"] for item in selected)),
        "gpt_selection_by_split": dict(Counter(item["parent_split"] for item in selected)),
        "kept_images": len(active_rows), "kept_positive": sum(row["positive"] == "1" for row in active_rows),
        "kept_negative": sum(row["positive"] == "0" for row in active_rows), "boxes": sum(int(row["box_count"]) for row in active_rows),
        "split_counts": {split: {"images": sum(row["split"] == split for row in active_rows), "positive": sum(row["split"] == split and row["positive"] == "1" for row in active_rows), "negative": sum(row["split"] == split and row["positive"] == "0" for row in active_rows), "boxes": sum(int(row["box_count"]) for row in active_rows if row["split"] == split)} for split in SPLITS},
        "source_counts": dict(Counter(row["source"] for row in active_rows)),
    }
    (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.rename(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
