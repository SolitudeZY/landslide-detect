#!/usr/bin/env python3
"""Export flagged positive images into review folders by audit reason."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


REASONS = (
    "portrait_domain_mismatch",
    "low_resolution",
    "possibly_blurred",
    "extreme_exposure",
    "tiny_target",
    "box_near_edge",
    "no_valid_box",
    "unreadable",
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    data_dir = project / "dataset/landslide_yolo_screened_2026-08-21"
    parser = argparse.ArgumentParser(description="按质量风险类型导出正样本复核目录")
    parser.add_argument("--audit", type=Path, default=data_dir / "positive_quality_audit.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "dataset/positive_quality_review_2026-08-26",
    )
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def put_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    audit = args.audit.resolve()
    output = args.output.resolve()
    if not audit.is_file():
        raise SystemExit(f"找不到审计结果: {audit}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"输出目录已存在且非空；如确认覆盖请加 --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with audit.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    flagged = [row for row in rows if row.get("reasons")]
    exported: dict[str, int] = Counter()
    all_seen: set[str] = set()
    index_rows: list[dict[str, str]] = []

    for row in flagged:
        source = Path(row["path"])
        if not source.is_file():
            row["export_error"] = "source_missing"
            index_rows.append(row)
            continue
        filename = source.name
        reasons = [reason for reason in row["reasons"].split(";") if reason]
        for reason in reasons:
            folder = reason if reason in REASONS else "other"
            put_file(source, output / folder / filename, args.mode)
            exported[folder] += 1
        if str(source) not in all_seen:
            put_file(source, output / "all_flagged" / filename, args.mode)
            all_seen.add(str(source))
        index_rows.append(row)

    index_path = output / "review_index.csv"
    fields = list(rows[0]) if rows else ["path", "reasons"]
    if "export_error" not in fields:
        fields.append("export_error")
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(index_rows)
    summary = {
        "audit": str(audit),
        "output": str(output),
        "mode": args.mode,
        "flagged_unique": len(flagged),
        "all_flagged_exported": len(all_seen),
        "exports_by_folder": dict(exported),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
