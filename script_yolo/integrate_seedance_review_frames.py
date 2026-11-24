#!/usr/bin/env python3
"""Extract reviewed Seedance frames and add normal frames as train negatives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2


NEGATIVE_FRAMES = (0, 12, 24, 36, 48, 60, 69)
POSITIVE_REVIEW_FRAMES = (
    78,
    84,
    90,
    96,
    102,
    108,
    114,
    120,
    126,
    132,
    138,
    144,
    150,
    156,
    162,
    168,
    174,
    180,
    186,
    192,
)
SOURCE = "seedance_ai_video_20260827"
GROUP = "video:seedance-landslide-test:scene:0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    os.replace(temporary, path)


def extract_frames(video: Path, targets: dict[int, Path]) -> tuple[float, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pending = set(targets)
    frame_index = 0
    try:
        while pending:
            ok, frame = capture.read()
            if not ok:
                break
            destination = targets.get(frame_index)
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.stem}.tmp.jpg")
                if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"无法写入图片: {temporary}")
                os.replace(temporary, destination)
                pending.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if pending:
        raise RuntimeError(f"视频缺少目标帧: {sorted(pending)}")
    return fps, width, height


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    video = project / "script_yolo/test-video-ai/seedance-landslide-test.mp4"
    data_dir = project / "dataset/landslide_yolo_gpt_clean_2026-08-26"
    negative_source_dir = project / "dataset/负样本-滑坡/seedance-ai-video-2026-08-27"
    positive_review_dir = project / "dataset/待标注-正样本-seedance-2026-08-27"
    positive_image_dir = positive_review_dir / "images"

    if not video.is_file():
        raise FileNotFoundError(video)
    manifest_path = data_dir / "manifest.csv"
    coco_path = data_dir / "annotations/instances_train.json"
    if not manifest_path.is_file() or not coco_path.is_file():
        raise FileNotFoundError("融合数据集缺少 manifest.csv 或 train COCO 标注")

    negative_targets = {
        frame: negative_source_dir / f"seedance_negative_frame_{frame:06d}.jpg"
        for frame in NEGATIVE_FRAMES
    }
    positive_targets = {
        frame: positive_image_dir / f"seedance_positive_frame_{frame:06d}.jpg"
        for frame in POSITIVE_REVIEW_FRAMES
    }
    fps, width, height = extract_frames(video, negative_targets | positive_targets)

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise ValueError(f"manifest 没有表头: {manifest_path}")

    existing_source_frames = {
        Path(row["original_path"]).name: row
        for row in rows
        if row["source"] == SOURCE and row["status"] == "kept"
    }
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco_names = {Path(item["file_name"]).name for item in coco["images"]}
    next_image_id = max((int(item["id"]) for item in coco["images"]), default=0) + 1

    added = 0
    for frame, source_path in negative_targets.items():
        source_name = source_path.name
        digest = sha256(source_path)
        output_name = f"{digest[:16]}_{SOURCE}.jpg"
        output_image = data_dir / "images/train" / output_name
        output_label = data_dir / "labels/train" / f"{Path(output_name).stem}.txt"

        existing = existing_source_frames.get(source_name)
        if existing is not None:
            expected = data_dir / existing["output_path"]
            if not expected.is_file() or existing["sha256"] != digest:
                raise RuntimeError(f"已有 Seedance 样本不完整或内容变化: {source_name}")
            continue
        if output_name in coco_names or output_image.exists() or output_label.exists():
            raise RuntimeError(f"输出文件名冲突: {output_name}")

        try:
            os.link(source_path, output_image)
        except OSError:
            shutil.copy2(source_path, output_image)
        output_label.write_text("", encoding="utf-8")
        rows.append(
            {
                "status": "kept",
                "source": SOURCE,
                "positive": "0",
                "split": "train",
                "group": GROUP,
                "original_path": str(source_path.resolve()),
                "output_path": f"images/train/{output_name}",
                "width": str(width),
                "height": str(height),
                "box_count": "0",
                "sha256": digest,
                "duplicate_of": "",
                "note": (
                    f"AI video hard negative; frame={frame}; "
                    f"timestamp={frame / fps:.3f}s; reviewed normal before landslide onset"
                ),
            }
        )
        coco["images"].append(
            {
                "id": next_image_id,
                "file_name": f"../images/train/{output_name}",
                "width": width,
                "height": height,
                "source": SOURCE,
            }
        )
        next_image_id += 1
        coco_names.add(output_name)
        added += 1

    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_manifest, manifest_path)
    write_atomic(coco_path, json.dumps(coco, ensure_ascii=False, indent=2))

    summary_path = data_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    active = [row for row in rows if row["status"] == "kept"]
    summary["kept_images"] = len(active)
    summary["kept_positive"] = sum(row["positive"] == "1" for row in active)
    summary["kept_negative"] = sum(row["positive"] == "0" for row in active)
    summary["boxes"] = sum(int(row["box_count"]) for row in active)
    summary["split_counts"] = {
        split: {
            "images": sum(row["split"] == split for row in active),
            "positive": sum(row["split"] == split and row["positive"] == "1" for row in active),
            "negative": sum(row["split"] == split and row["positive"] == "0" for row in active),
            "boxes": sum(int(row["box_count"]) for row in active if row["split"] == split),
        }
        for split in ("train", "val", "test")
    }
    summary["source_counts"] = dict(Counter(row["source"] for row in active))
    write_atomic(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))

    review_rows = [
        {
            "filename": path.name,
            "frame": frame,
            "timestamp_seconds": f"{frame / fps:.3f}",
            "width": width,
            "height": height,
            "suggested_phase": (
                "onset" if frame <= 90 else "active" if frame <= 156 else "dust_and_deposition"
            ),
            "source_video": str(video.resolve()),
        }
        for frame, path in positive_targets.items()
    ]
    review_manifest = positive_review_dir / "review_manifest.csv"
    temporary_review = review_manifest.with_name(f".{review_manifest.name}.tmp")
    with temporary_review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    os.replace(temporary_review, review_manifest)
    write_atomic(
        positive_review_dir / "README.md",
        "# Seedance 滑坡正样本待标注\n\n"
        "图片来自同一条 AI 生成视频的滑坡阶段，尚未加入训练数据集。请只框选实际滑动的土体、落石和相关扬尘区域，避免用框覆盖整面静态山体或挡墙。\n\n"
        "标注完成后，这批正样本必须与已加入 train 的同视频负样本一起归入 train，不能分配到 val/test。\n",
    )

    print(
        json.dumps(
            {
                "video": str(video),
                "fps": fps,
                "negative_frames": len(negative_targets),
                "negative_added_to_train": added,
                "positive_review_frames": len(positive_targets),
                "positive_review_dir": str(positive_review_dir),
                "dataset": str(data_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
