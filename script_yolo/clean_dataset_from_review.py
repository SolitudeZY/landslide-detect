#!/usr/bin/env python3
"""Remove manually rejected audited positives from a fused YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    dataset = project / "dataset/landslide_yolo_screened_2026-08-21"
    review = project / "dataset/positive_quality_review_2026-08-26"
    parser = argparse.ArgumentParser(description="按人工复核清单清理融合 YOLO 数据集")
    parser.add_argument("--data-dir", type=Path, default=dataset)
    parser.add_argument("--review-index", type=Path, default=review / "review_index.csv")
    parser.add_argument("--keep-index", type=Path, default=review / "review_index_filtered.csv")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def backup(path: Path, backup_dir: Path) -> None:
    if path.is_file():
        target = backup_dir / path.relative_to(path.parents[2])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    review_rows = read_csv(args.review_index.resolve())
    keep_rows = read_csv(args.keep_index.resolve())
    audit_names = {Path(row["path"]).name for row in review_rows}
    keep_names = {Path(row["path"]).name for row in keep_rows}
    if len(audit_names) != len(review_rows) or len(keep_names) != len(keep_rows):
        raise SystemExit("复核清单存在重复文件名，拒绝继续")
    deleted_names = audit_names - keep_names
    if not deleted_names:
        print("没有需要删除的图片，将仅刷新数据集统计和索引")

    manifest_path = data_dir / "manifest.csv"
    manifest_rows = read_csv(manifest_path)
    target_rows = [
        row for row in manifest_rows
        if Path(row.get("output_path", "")).name in deleted_names
    ]
    target_names = {Path(row["output_path"]).name for row in target_rows}
    if target_names != deleted_names:
        missing = sorted(deleted_names - target_names)
        extra = sorted(target_names - deleted_names)
        raise SystemExit(f"训练集与审核清单不一致: missing={missing[:5]} extra={extra[:5]}")

    files_to_delete: list[Path] = []
    for row in target_rows:
        image = data_dir / row["output_path"]
        label = data_dir / "labels" / row["split"] / f"{image.stem}.txt"
        files_to_delete.extend(path for path in (image, label) if path.is_file())
    if args.dry_run:
        print(json.dumps({"deleted_images": len(deleted_names), "files": len(files_to_delete)}, ensure_ascii=False))
        return

    backup_dir = (args.backup_dir or data_dir / ".cleanup_backup_2026-08-26").resolve()
    metadata = [
        manifest_path,
        data_dir / "summary.json",
        data_dir / "calibration.txt",
        data_dir / "calibration_wsl.txt",
        data_dir / "domain_balanced_train.txt",
        data_dir / "domain_balanced_train_summary.json",
    ] + list((data_dir / "annotations").glob("instances_*.json"))
    for path in metadata:
        backup(path, backup_dir)

    for path in files_to_delete:
        path.unlink()

    for row in manifest_rows:
        if Path(row.get("output_path", "")).name in deleted_names and row.get("status") == "kept":
            row["status"] = "excluded_quality_review"
            row["note"] = "人工质量复核后删除"
    fields = list(manifest_rows[0])
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest_rows)

    for annotation_path in data_dir.joinpath("annotations").glob("instances_*.json"):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        kept_images = [item for item in data.get("images", []) if Path(item["file_name"]).name not in deleted_names]
        kept_ids = {item["id"] for item in kept_images}
        data["images"] = kept_images
        data["annotations"] = [item for item in data.get("annotations", []) if item["image_id"] in kept_ids]
        annotation_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for list_name in ("calibration.txt", "calibration_wsl.txt", "domain_balanced_train.txt"):
        path = data_dir / list_name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line for line in lines if Path(line).name not in deleted_names) + "\n", encoding="utf-8")

    for cache in (data_dir / "labels").glob("*.cache"):
        cache.unlink()

    active = [row for row in manifest_rows if row.get("status") == "kept"]
    summary_path = data_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["kept_images"] = len(active)
    summary["kept_positive"] = sum(row["positive"] == "1" for row in active)
    summary["kept_negative"] = sum(row["positive"] == "0" for row in active)
    summary["boxes"] = sum(int(row.get("box_count", 0)) for row in active)
    summary["status_counts"] = dict(Counter(row["status"] for row in manifest_rows))
    summary["source_counts"] = dict(Counter(row["source"] for row in active))
    summary["source_split_counts"] = {
        source: dict(Counter(row["split"] for row in active if row["source"] == source))
        for source in sorted({row["source"] for row in active})
    }
    summary["split_counts"] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in active if row["split"] == split]
        summary["split_counts"][split] = {
            "images": len(rows),
            "positive": sum(row["positive"] == "1" for row in rows),
            "negative": sum(row["positive"] == "0" for row in rows),
            "boxes": sum(int(row.get("box_count", 0)) for row in rows),
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "deleted_unique_images": len(deleted_names),
        "deleted_files": len(files_to_delete),
        "backup_dir": str(backup_dir),
        "remaining_kept": len(active),
        "remaining_positive": summary["kept_positive"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
