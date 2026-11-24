#!/usr/bin/env python3
"""Merge reviewed COCO positives and negative images into a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import imagehash
import yaml
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    source: str
    path: Path
    positive: bool
    boxes: list[list[float]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    sha256: str = ""
    phash: int | None = None
    phash_flipped: int | None = None
    status: str = "candidate"
    note: str = ""
    duplicate_of: str = ""
    group: str = ""
    split: str = ""
    output_name: str = ""


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent / "dataset"
    parser = argparse.ArgumentParser(
        description="融合审核后的 COCO 正样本和无标注负样本，输出 YOLO 检测数据集。"
    )
    parser.add_argument(
        "--reviewed-coco",
        type=Path,
        default=dataset_dir
        / "正样本-滑坡/landslide-26-08-19/annotations/instances_Train.json",
    )
    parser.add_argument(
        "--reviewed-images",
        type=Path,
        default=dataset_dir / "正样本-滑坡/landslide-26-08-19/images/Train",
    )
    parser.add_argument(
        "--supplement-coco",
        type=Path,
        default=dataset_dir
        / "正样本-滑坡/landslide补充-v1-08-19/annotations/instances_Train.json",
    )
    parser.add_argument(
        "--supplement-images",
        type=Path,
        default=dataset_dir / "正样本-滑坡/landslide补充-v1-08-19/images/Train",
    )
    parser.add_argument(
        "--new-coco",
        type=Path,
        default=dataset_dir
        / "正样本-滑坡/landslide-08-21/annotations/instances_Validation.json",
    )
    parser.add_argument(
        "--new-images",
        type=Path,
        default=dataset_dir / "正样本-滑坡/landslide-08-21/images/Validation",
    )
    parser.add_argument(
        "--negative-dir",
        action="append",
        type=Path,
        dest="negative_dirs",
        help="可重复传入；未指定时使用项目内已审核的负样本目录。",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=["landslide_3_"],
        help="按文件名前缀整组排除；可重复传入。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_dir / "landslide_yolo_screened_2026-08-21",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="INT8 校准图片总数；从 train 中按正负比例分层抽样。",
    )
    parser.add_argument(
        "--calibration-positive-ratio",
        type=float,
        default=0.5,
        help="INT8 校准集中正样本比例，默认 0.5。",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="仅用于 reviewed 来源；0 表示关闭近重复去重。",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.negative_dirs is None:
        args.negative_dirs = [
            dataset_dir / "负样本-滑坡/captures_v2_reviewed_negative",
            dataset_dir / "负样本-滑坡/negative_samples",
            dataset_dir / "负样本-滑坡/26-8-18-real",
            dataset_dir / "负样本-滑坡/26_08_19-21-real",
            dataset_dir / "待复核-08-21-无框样本/images",
        ]
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("train/val/test 比例必须非负且总和为 1。")
    if not 0 <= args.phash_threshold <= 16:
        parser.error("phash-threshold 必须在 0 到 16 之间。")
    if args.calibration_size <= 0:
        parser.error("calibration-size 必须大于 0。")
    if not 0 <= args.calibration_positive_ratio <= 1:
        parser.error("calibration-positive-ratio 必须在 0 到 1 之间。")
    return args


def select_calibration_samples(
    samples: list[Sample], total: int, positive_ratio: float, seed: int
) -> list[Sample]:
    positives = [item for item in samples if item.split == "train" and item.positive]
    negatives = [item for item in samples if item.split == "train" and not item.positive]
    positive_count = round(total * positive_ratio)
    negative_count = total - positive_count
    if len(positives) < positive_count:
        raise ValueError(
            f"校准集正样本不足：需要 {positive_count}，只有 {len(positives)}"
        )
    if len(negatives) < negative_count:
        raise ValueError(
            f"校准集负样本不足：需要 {negative_count}，只有 {len(negatives)}"
        )
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected = positives[:positive_count] + negatives[:negative_count]
    rng.shuffle(selected)
    return selected


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt" or not resolved.drive:
        return resolved.as_posix()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{relative}"


def load_coco(source: str, json_path: Path, image_dir: Path) -> list[Sample]:
    if not json_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"COCO 输入不存在: {json_path} / {image_dir}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in data.get("categories", [])}
    if not categories or any(name.lower() != "landslide" for name in categories.values()):
        raise ValueError(f"{json_path} 的类别不是单一 landslide: {categories}")

    annotations: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        if annotation.get("category_id") not in categories:
            raise ValueError(f"未知 category_id: {annotation}")
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            raise ValueError(f"非法 COCO bbox: {annotation}")
        annotations[int(annotation["image_id"])].append([float(value) for value in bbox])

    samples: list[Sample] = []
    seen_image_ids: set[int] = set()
    for image in data.get("images", []):
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            raise ValueError(f"重复 COCO image id: {image_id}")
        seen_image_ids.add(image_id)
        path = image_dir / image["file_name"]
        sample = Sample(
            source=source,
            path=path,
            positive=True,
            boxes=annotations.get(image_id, []),
            width=int(image.get("width", 0)),
            height=int(image.get("height", 0)),
        )
        if not sample.boxes:
            sample.status = "excluded_no_box"
            sample.note = "COCO 正样本没有任何矩形框"
        samples.append(sample)
    unknown_image_ids = set(annotations) - seen_image_ids
    if unknown_image_ids:
        raise ValueError(f"标注引用了不存在的 image id: {sorted(unknown_image_ids)[:10]}")
    return samples


def negative_source(directory: Path, index: int) -> str:
    sources = {
        "captures_v2_reviewed_negative": "negative_captures",
        "negative_samples": "negative_video_2",
        "26-8-18-real": "negative_video_3",
        "26_08_19-21-real": "negative_20260821",
    }
    if "待复核-08-21-无框样本" in str(directory):
        return "reviewed_negative_20260821"
    return sources.get(directory.name, f"negative_video_{index}")


def load_negatives(negative_dirs: Iterable[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for index, directory in enumerate(negative_dirs, start=1):
        if not directory.is_dir():
            raise FileNotFoundError(f"负样本目录不存在: {directory}")
        source = negative_source(directory, index)
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(Sample(source=source, path=path, positive=False))
    return samples


def exclude_samples(samples: list[Sample], prefixes: Iterable[str]) -> None:
    prefixes = tuple(prefixes)
    for sample in samples:
        if sample.path.name.startswith(prefixes):
            sample.status = "excluded_user"
            sample.note = "用户审核后整组排除"


def inspect_images(samples: list[Sample]) -> None:
    for index, sample in enumerate(samples, start=1):
        if sample.status != "candidate":
            continue
        if not sample.path.is_file():
            sample.status = "excluded_missing"
            sample.note = "图片文件不存在"
            continue
        try:
            digest = hashlib.sha256()
            with sample.path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            sample.sha256 = digest.hexdigest()
            with Image.open(sample.path) as image:
                image.load()
                actual_width, actual_height = image.size
                if sample.width and sample.height and (sample.width, sample.height) != image.size:
                    sample.note = (
                        f"COCO 尺寸 {sample.width}x{sample.height} 已修正为 "
                        f"{actual_width}x{actual_height}"
                    )
                sample.width, sample.height = actual_width, actual_height
                if sample.source == "reviewed":
                    sample.phash = int(str(imagehash.phash(ImageOps.grayscale(image))), 16)
                    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    sample.phash_flipped = int(
                        str(imagehash.phash(ImageOps.grayscale(flipped))), 16
                    )
        except (OSError, ValueError) as exc:
            sample.status = "excluded_unreadable"
            sample.note = f"图片无法读取: {exc}"
        if index % 500 == 0:
            print(f"已检查 {index}/{len(samples)} 张图片", file=sys.stderr)


def clip_boxes(samples: list[Sample]) -> None:
    for sample in samples:
        if sample.status != "candidate" or not sample.positive:
            continue
        clipped: list[list[float]] = []
        invalid = 0
        for x, y, width, height in sample.boxes:
            x1 = max(0.0, min(float(sample.width), x))
            y1 = max(0.0, min(float(sample.height), y))
            x2 = max(0.0, min(float(sample.width), x + width))
            y2 = max(0.0, min(float(sample.height), y + height))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                invalid += 1
                continue
            clipped.append([x1, y1, x2 - x1, y2 - y1])
        sample.boxes = clipped
        if invalid:
            sample.note = "; ".join(filter(None, (sample.note, f"剔除 {invalid} 个非法框")))
        if not sample.boxes:
            sample.status = "excluded_no_valid_box"
            sample.note = "; ".join(filter(None, (sample.note, "没有有效矩形框")))


def exact_deduplicate(samples: list[Sample], duplicates: list[dict[str, object]]) -> None:
    priority = {"new_positive_20260821": 4, "supplement": 3, "reviewed": 2}
    candidates = sorted(
        (sample for sample in samples if sample.status == "candidate"),
        key=lambda item: (-priority.get(item.source, 0), str(item.path)),
    )
    seen: dict[str, Sample] = {}
    for sample in candidates:
        kept = seen.get(sample.sha256)
        if kept is None:
            seen[sample.sha256] = sample
            continue
        if kept.positive != sample.positive:
            raise ValueError(
                "同一图片同时出现在正负样本中，请人工复核: "
                f"{kept.path} <-> {sample.path}"
            )
        sample.status = "excluded_exact_duplicate"
        sample.duplicate_of = str(kept.path)
        duplicates.append(
            {
                "reason": "exact_sha256",
                "distance": 0,
                "kept_path": str(kept.path),
                "duplicate_path": str(sample.path),
            }
        )


def perceptual_deduplicate(
    samples: list[Sample], threshold: int, duplicates: list[dict[str, object]]
) -> None:
    reviewed = [
        sample
        for sample in samples
        if sample.status == "candidate" and sample.source == "reviewed"
    ]
    if threshold <= 0 or len(reviewed) < 2:
        for sample in reviewed:
            sample.group = f"reviewed:{sample.sha256[:16]}"
        return

    sets = DisjointSet(len(reviewed))
    for left in range(len(reviewed)):
        left_hash = reviewed[left].phash
        if left_hash is None:
            continue
        for right in range(left + 1, len(reviewed)):
            normal = reviewed[right].phash
            flipped = reviewed[right].phash_flipped
            if normal is None or flipped is None:
                continue
            distance = min((left_hash ^ normal).bit_count(), (left_hash ^ flipped).bit_count())
            if distance <= threshold:
                sets.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(reviewed)):
        clusters[sets.find(index)].append(index)
    for member_indices in clusters.values():
        members = [reviewed[index] for index in member_indices]
        kept = max(
            members,
            key=lambda item: (item.width * item.height, len(item.boxes), -len(str(item.path))),
        )
        cluster_id = kept.sha256[:16]
        kept.group = f"reviewed:{cluster_id}"
        for index in member_indices:
            sample = reviewed[index]
            sample.group = f"reviewed:{cluster_id}"
            if sample is kept:
                continue
            distance = min(
                (kept.phash ^ sample.phash).bit_count(),
                (kept.phash ^ sample.phash_flipped).bit_count(),
            )
            sample.status = "excluded_near_duplicate"
            sample.duplicate_of = str(kept.path)
            duplicates.append(
                {
                    "reason": "phash_or_horizontal_flip",
                    "distance": distance,
                    "kept_path": str(kept.path),
                    "duplicate_path": str(sample.path),
                }
            )


def source_group(sample: Sample) -> str:
    if sample.group:
        return sample.group
    return f"{sample.source}:single:{sample.sha256[:16]}"


def assign_temporal_groups(samples: list[Sample]) -> None:
    timelines: dict[str, list[tuple[float, Sample]]] = defaultdict(list)
    for sample in samples:
        if sample.group:
            continue
        name = sample.path.name
        video_match = re.search(r"\.mp4_(\d{6})\.(\d+)", name)
        if video_match:
            seconds = int(video_match.group(1)) + float(f"0.{video_match.group(2)}")
            prefix = name[: video_match.start()]
            origin = f"video:{hashlib.sha1(prefix.encode()).hexdigest()[:10]}"
            timelines[origin].append((seconds, sample))
            continue
        sichuan_match = re.match(r"sichuan-landslide_(\d+)_", name)
        if sichuan_match:
            timelines["video:sichuan-landslide"].append(
                (int(sichuan_match.group(1)) / 1000.0, sample)
            )
            continue
        vlcsnap_match = re.match(
            r"vlcsnap-(\d{4}-\d{2}-\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d+)",
            name,
        )
        if vlcsnap_match:
            date, hour, minute, second, fraction = vlcsnap_match.groups()
            timestamp = (
                int(hour) * 3600
                + int(minute) * 60
                + int(second)
                + float(f"0.{fraction}")
            )
            timelines[f"{sample.source}:vlcsnap:{date}"].append((timestamp, sample))
            continue
        sequence_match = re.match(r"(.+)_([0-9]{4})\.[^.]+$", name)
        if sequence_match:
            prefix, frame_number = sequence_match.groups()
            timelines[f"{sample.source}:sequence:{prefix}"].append(
                (int(frame_number), sample)
            )
            continue
        frame_match = re.match(r"(?:video)?frame_(\d+)", name)
        if frame_match:
            parent_id = hashlib.sha1(str(sample.path.parent).encode()).hexdigest()[:10]
            timelines[f"{sample.source}:frames:{parent_id}"].append(
                (int(frame_match.group(1)) / 30.0, sample)
            )
            continue
        try:
            epoch = float(sample.path.stem)
        except ValueError:
            continue
        timelines[f"{sample.source}:captures"].append((epoch, sample))

    for origin, entries in timelines.items():
        entries.sort(key=lambda item: item[0])
        is_capture = origin.endswith(":captures") or ":vlcsnap:" in origin
        is_frame_sequence = ":frames:" in origin or ":sequence:" in origin
        max_gap = 60.0 if is_capture else (float("inf") if is_frame_sequence else 3.0)
        scene = 0
        previous_time: float | None = None
        for timestamp, sample in entries:
            if previous_time is not None and timestamp - previous_time > max_gap:
                scene += 1
            sample.group = f"{origin}:scene:{scene}"
            previous_time = timestamp


def assign_splits(samples: list[Sample], ratios: dict[str, float], seed: int) -> None:
    active = [sample for sample in samples if sample.status == "candidate"]
    assign_temporal_groups(active)
    global_groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in active:
        sample.group = source_group(sample)
        global_groups[sample.group].append(sample)

    strata: dict[tuple[tuple[str, ...], bool, bool], dict[str, list[Sample]]] = defaultdict(dict)
    for group_name, group_samples in global_groups.items():
        stratum = (
            tuple(sorted({sample.source for sample in group_samples})),
            any(sample.positive for sample in group_samples),
            any(not sample.positive for sample in group_samples),
        )
        strata[stratum][group_name] = group_samples

    for stratum, grouped in sorted(strata.items(), key=lambda item: str(item[0])):
        rng = random.Random(f"{seed}:{stratum}")
        weighted_groups = list(grouped.items())
        rng.shuffle(weighted_groups)
        weighted_groups.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(group) for _, group in weighted_groups)
        targets = {split: max(ratios[split] * total, 1e-9) for split in SPLITS}
        counts = {split: 0 for split in SPLITS}
        for _, group_samples in weighted_groups:
            split = min(
                SPLITS,
                key=lambda name: ((counts[name] + len(group_samples)) / targets[name], SPLITS.index(name)),
            )
            for sample in group_samples:
                sample.split = split
            counts[split] += len(group_samples)


def yolo_lines(sample: Sample) -> list[str]:
    lines = []
    for x, y, width, height in sample.boxes:
        center_x = (x + width / 2.0) / sample.width
        center_y = (y + height / 2.0) / sample.height
        lines.append(
            f"0 {center_x:.8f} {center_y:.8f} "
            f"{width / sample.width:.8f} {height / sample.height:.8f}"
        )
    return lines


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source.resolve())


def write_outputs(
    samples: list[Sample],
    duplicates: list[dict[str, object]],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    active = [sample for sample in samples if sample.status == "candidate"]
    staging = output.with_name(f".{output.name}.tmp")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在；确认后使用 --overwrite: {output}")
    if staging.exists():
        shutil.rmtree(staging)
    for split in SPLITS:
        (staging / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
    (staging / "annotations").mkdir(parents=True, exist_ok=True)

    coco_by_split = {
        split: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "landslide"}]}
        for split in SPLITS
    }
    image_ids = {split: 1 for split in SPLITS}
    annotation_ids = {split: 1 for split in SPLITS}
    used_names: set[str] = set()
    for sample in sorted(active, key=lambda item: (item.split, item.source, str(item.path))):
        extension = sample.path.suffix.lower()
        prefix = re.sub(r"[^a-z0-9]+", "_", sample.source.lower()).strip("_")
        output_name = f"{sample.sha256[:16]}_{prefix}{extension}"
        if output_name in used_names:
            raise RuntimeError(f"输出文件名冲突: {output_name}")
        used_names.add(output_name)
        sample.output_name = output_name
        image_destination = staging / "images" / sample.split / output_name
        label_destination = staging / "labels" / sample.split / f"{Path(output_name).stem}.txt"
        transfer_file(sample.path, image_destination, args.copy_mode)
        label_destination.write_text("\n".join(yolo_lines(sample)) + ("\n" if sample.boxes else ""))

        split_coco = coco_by_split[sample.split]
        image_id = image_ids[sample.split]
        image_ids[sample.split] += 1
        split_coco["images"].append(
            {
                "id": image_id,
                "file_name": f"../images/{sample.split}/{output_name}",
                "width": sample.width,
                "height": sample.height,
                "source": sample.source,
            }
        )
        for box in sample.boxes:
            annotation_id = annotation_ids[sample.split]
            annotation_ids[sample.split] += 1
            split_coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )

    for split, coco in coco_by_split.items():
        (staging / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "landslide"},
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    manifest_fields = [
        "status",
        "source",
        "positive",
        "split",
        "group",
        "original_path",
        "output_path",
        "width",
        "height",
        "box_count",
        "sha256",
        "duplicate_of",
        "note",
    ]
    with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: (item.status, item.source, str(item.path))):
            writer.writerow(
                {
                    "status": "kept" if sample.status == "candidate" else sample.status,
                    "source": sample.source,
                    "positive": int(sample.positive),
                    "split": sample.split,
                    "group": sample.group,
                    "original_path": str(sample.path.resolve()),
                    "output_path": (
                        f"images/{sample.split}/{sample.output_name}" if sample.output_name else ""
                    ),
                    "width": sample.width,
                    "height": sample.height,
                    "box_count": len(sample.boxes),
                    "sha256": sample.sha256,
                    "duplicate_of": sample.duplicate_of,
                    "note": sample.note,
                }
            )

    with (staging / "duplicates.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["reason", "distance", "kept_path", "duplicate_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicates)

    calibration = select_calibration_samples(
        active,
        args.calibration_size,
        args.calibration_positive_ratio,
        args.seed,
    )
    calibration_paths = [
        output / "images" / item.split / item.output_name for item in calibration
    ]
    (staging / "calibration.txt").write_text(
        "\n".join(str(path.resolve()) for path in calibration_paths)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (staging / "calibration_wsl.txt").write_text(
        "\n".join(wsl_path(path) for path in calibration_paths)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary: dict[str, object] = {
        "seed": args.seed,
        "ratios": {split: getattr(args, f"{split}_ratio") for split in SPLITS},
        "phash_threshold": args.phash_threshold,
        "input_images": len(samples),
        "kept_images": len(active),
        "kept_positive": sum(item.positive for item in active),
        "kept_negative": sum(not item.positive for item in active),
        "boxes": sum(len(item.boxes) for item in active),
        "calibration": {
            "images": len(calibration),
            "positive": sum(item.positive for item in calibration),
            "negative": sum(not item.positive for item in calibration),
        },
        "status_counts": dict(Counter("kept" if item.status == "candidate" else item.status for item in samples)),
        "split_counts": {
            split: {
                "images": sum(item.split == split for item in active),
                "positive": sum(item.split == split and item.positive for item in active),
                "negative": sum(item.split == split and not item.positive for item in active),
                "boxes": sum(len(item.boxes) for item in active if item.split == split),
            }
            for split in SPLITS
        },
        "source_counts": dict(Counter(item.source for item in active)),
        "source_split_counts": {
            source: {
                split: sum(item.source == source and item.split == split for item in active)
                for split in SPLITS
            }
            for source in sorted({item.source for item in active})
        },
        "near_duplicate_clusters": duplicate_cluster_summary(samples),
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return summary


def preview_summary(samples: list[Sample], args: argparse.Namespace) -> dict[str, object]:
    active = [sample for sample in samples if sample.status == "candidate"]
    return {
        "dry_run": True,
        "output": str(args.output.resolve()),
        "input_images": len(samples),
        "kept_images": len(active),
        "kept_positive": sum(item.positive for item in active),
        "kept_negative": sum(not item.positive for item in active),
        "boxes": sum(len(item.boxes) for item in active),
        "status_counts": dict(Counter("kept" if item.status == "candidate" else item.status for item in samples)),
        "near_duplicate_clusters": duplicate_cluster_summary(samples),
        "split_counts": {
            split: {
                "images": sum(item.split == split for item in active),
                "positive": sum(item.split == split and item.positive for item in active),
                "negative": sum(item.split == split and not item.positive for item in active),
            }
            for split in SPLITS
        },
        "source_split_counts": {
            source: {
                split: sum(item.source == source and item.split == split for item in active)
                for split in SPLITS
            }
            for source in sorted({item.source for item in active})
        },
    }


def duplicate_cluster_summary(samples: list[Sample]) -> dict[str, int]:
    sizes = Counter(
        sample.group
        for sample in samples
        if sample.source == "reviewed" and sample.group
    )
    duplicate_sizes = [size for size in sizes.values() if size > 1]
    return {
        "clusters": len(duplicate_sizes),
        "largest_cluster": max(duplicate_sizes, default=1),
        "images_in_clusters": sum(duplicate_sizes),
    }


def main() -> None:
    args = parse_args()
    samples = load_coco("reviewed", args.reviewed_coco, args.reviewed_images)
    samples.extend(load_coco("supplement", args.supplement_coco, args.supplement_images))
    samples.extend(load_coco("new_positive_20260821", args.new_coco, args.new_images))
    samples.extend(load_negatives(args.negative_dirs))
    exclude_samples(samples, args.exclude_prefix)
    duplicates: list[dict[str, object]] = []
    inspect_images(samples)
    clip_boxes(samples)
    exact_deduplicate(samples, duplicates)
    perceptual_deduplicate(samples, args.phash_threshold, duplicates)
    assign_splits(
        samples,
        {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        args.seed,
    )
    summary = (
        preview_summary(samples, args)
        if args.dry_run
        else write_outputs(samples, duplicates, args.output, args)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
