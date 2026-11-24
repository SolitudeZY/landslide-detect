#!/usr/bin/env python3
"""Integrate manually annotated Seedance frames into the fused train split."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SOURCE = "seedance_positive_20260827"
GROUP = "video:seedance-landslide-test:scene:0"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def update_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    active = [row for row in rows if row["status"] == "kept"]
    summary["kept_images"] = len(active)
    summary["kept_positive"] = sum(row["positive"] == "1" for row in active)
    summary["kept_negative"] = sum(row["positive"] == "0" for row in active)
    summary["boxes"] = sum(int(row["box_count"]) for row in active)
    summary["split_counts"] = {
        split: {
            "images": sum(row["split"] == split for row in active),
            "positive": sum(
                row["split"] == split and row["positive"] == "1" for row in active
            ),
            "negative": sum(
                row["split"] == split and row["positive"] == "0" for row in active
            ),
            "boxes": sum(
                int(row["box_count"])
                for row in active
                if row["split"] == split
            ),
        }
        for split in ("train", "val", "test")
    }
    summary["source_counts"] = dict(Counter(row["source"] for row in active))
    write_json_atomic(path, summary)


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    export_dir = project / "dataset/seedance-landslide-08-27"
    annotation_path = export_dir / "annotations/instances_Train.json"
    image_dir = export_dir / "images/Train"
    data_dir = project / "dataset/landslide_yolo_gpt_clean_2026-08-26"
    manifest_path = data_dir / "manifest.csv"
    output_coco_path = data_dir / "annotations/instances_train.json"

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in coco.get("categories", [])}
    if set(categories.values()) != {"landslide"}:
        raise ValueError(f"类别必须只有 landslide，实际为: {categories}")

    images = {int(item["id"]): item for item in coco.get("images", [])}
    annotations: dict[int, list[list[float]]] = defaultdict(list)
    for item in coco.get("annotations", []):
        image_id = int(item["image_id"])
        if image_id not in images:
            raise ValueError(f"标注引用未知 image_id: {image_id}")
        if categories.get(int(item["category_id"])) != "landslide":
            raise ValueError(f"标注类别不是 landslide: {item}")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"标注缺少 COCO bbox: {item}")
        annotations[image_id].append([float(value) for value in box])
    if len(images) != 20:
        raise ValueError(f"预期20张 Seedance 图片，实际为 {len(images)}")
    missing = [item["file_name"] for image_id, item in images.items() if not annotations[image_id]]
    if missing:
        raise ValueError(f"以下图片没有滑坡框，拒绝作为正样本: {missing}")

    samples = []
    for image_id, item in sorted(images.items()):
        source_path = image_dir / item["file_name"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as image:
            image.load()
            width, height = image.size
        if (width, height) != (int(item["width"]), int(item["height"])):
            raise ValueError(f"COCO 尺寸与图片不一致: {source_path}")
        boxes = annotations[image_id]
        for x, y, box_width, box_height in boxes:
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ValueError(f"bbox 越界或无效: {source_path}: {[x, y, box_width, box_height]}")
        samples.append(
            {
                "source_path": source_path,
                "width": width,
                "height": height,
                "boxes": boxes,
                "sha256": digest(source_path),
            }
        )

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise ValueError(f"manifest 没有表头: {manifest_path}")
    existing_hashes = {
        row["sha256"]: row for row in rows if row["status"] == "kept"
    }
    existing_source = {
        Path(row["original_path"]).name: row
        for row in rows
        if row["status"] == "kept" and row["source"] == SOURCE
    }

    output_coco = json.loads(output_coco_path.read_text(encoding="utf-8"))
    coco_names = {Path(item["file_name"]).name for item in output_coco["images"]}
    next_image_id = max((int(item["id"]) for item in output_coco["images"]), default=0) + 1
    next_annotation_id = max(
        (int(item["id"]) for item in output_coco["annotations"]), default=0
    ) + 1

    added = 0
    for sample in samples:
        source_path = sample["source_path"]
        previous = existing_source.get(source_path.name)
        if previous is not None:
            if previous["sha256"] != sample["sha256"] or previous["box_count"] != str(len(sample["boxes"])):
                raise RuntimeError(f"已有 Seedance 正样本与本次导出不一致: {source_path.name}")
            continue
        duplicate = existing_hashes.get(sample["sha256"])
        if duplicate is not None:
            raise RuntimeError(
                f"图片已作为其他样本存在，拒绝重复加入: {source_path} -> {duplicate['output_path']}"
            )

        output_name = f"{sample['sha256'][:16]}_{SOURCE}.jpg"
        if output_name in coco_names:
            raise RuntimeError(f"COCO 输出名称冲突: {output_name}")
        output_image = data_dir / "images/train" / output_name
        output_label = data_dir / "labels/train" / f"{Path(output_name).stem}.txt"
        if output_image.exists() or output_label.exists():
            raise RuntimeError(f"输出文件冲突: {output_name}")

        try:
            os.link(source_path, output_image)
        except OSError:
            shutil.copy2(source_path, output_image)
        yolo_lines = []
        for x, y, box_width, box_height in sample["boxes"]:
            yolo_lines.append(
                "0 "
                f"{(x + box_width / 2) / sample['width']:.8f} "
                f"{(y + box_height / 2) / sample['height']:.8f} "
                f"{box_width / sample['width']:.8f} "
                f"{box_height / sample['height']:.8f}"
            )
        output_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        rows.append(
            {
                "status": "kept",
                "source": SOURCE,
                "positive": "1",
                "split": "train",
                "group": GROUP,
                "original_path": str(source_path.resolve()),
                "output_path": f"images/train/{output_name}",
                "width": str(sample["width"]),
                "height": str(sample["height"]),
                "box_count": str(len(sample["boxes"])),
                "sha256": sample["sha256"],
                "duplicate_of": "",
                "note": "AI video frame; manually annotated; same-video hard negatives are in train",
            }
        )
        output_coco["images"].append(
            {
                "id": next_image_id,
                "file_name": f"../images/train/{output_name}",
                "width": sample["width"],
                "height": sample["height"],
                "source": SOURCE,
            }
        )
        for box in sample["boxes"]:
            output_coco["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            next_annotation_id += 1
        next_image_id += 1
        coco_names.add(output_name)
        existing_hashes[sample["sha256"]] = rows[-1]
        added += 1

    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_manifest, manifest_path)
    write_json_atomic(output_coco_path, output_coco)
    update_summary(data_dir / "summary.json", rows)

    print(
        json.dumps(
            {
                "annotation": str(annotation_path),
                "validated_images": len(samples),
                "validated_boxes": sum(len(sample["boxes"]) for sample in samples),
                "added_to_train": added,
                "source": SOURCE,
                "group": GROUP,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
