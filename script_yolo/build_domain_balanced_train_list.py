#!/usr/bin/env python3
"""Build a source-balanced training list without copying dataset images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


DOMAIN_POSITIVE_SOURCES = ("supplement", "new_positive_20260821")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "dataset/landslide_yolo_screened_2026-08-21"
    parser = argparse.ArgumentParser(
        description="按来源和事件组生成监控域均衡的 YOLO 训练清单。"
    )
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--web-positive-limit", type=int, default=500)
    parser.add_argument("--domain-group-cap", type=int, default=16)
    parser.add_argument("--domain-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output-list",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.web_positive_limit <= 0:
        parser.error("web-positive-limit 必须大于 0")
    if args.domain_group_cap <= 0:
        parser.error("domain-group-cap 必须大于 0")
    if args.domain_repeat <= 0:
        parser.error("domain-repeat 必须大于 0")
    if args.output_list is None:
        args.output_list = args.data_dir / "domain_balanced_train.txt"
    if args.output_yaml is None:
        args.output_yaml = args.data_dir / "data_domain_balanced.yaml"
    if args.output_summary is None:
        args.output_summary = args.data_dir / "domain_balanced_train_summary.json"
    return args


def load_training_rows(data_dir: Path) -> list[dict[str, str]]:
    manifest = data_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "kept" and row["split"] == "train"
        ]
    if not rows:
        raise ValueError(f"训练清单为空: {manifest}")
    return rows


def image_path(data_dir: Path, row: dict[str, str]) -> Path:
    path = (data_dir / row["output_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evenly_spaced(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: row["original_path"])
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    rows = load_training_rows(data_dir)
    rng = random.Random(args.seed)

    negatives = [row for row in rows if row["positive"] == "0"]
    web_positives = [
        row for row in rows if row["positive"] == "1" and row["source"] == "reviewed"
    ]
    rng.shuffle(web_positives)
    selected_web = web_positives[: args.web_positive_limit]

    domain_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["positive"] == "1" and row["source"] in DOMAIN_POSITIVE_SOURCES:
            domain_groups[(row["source"], row["group"])].append(row)
    selected_domain = [
        row
        for group_rows in domain_groups.values()
        for row in evenly_spaced(group_rows, args.domain_group_cap)
    ]

    known_positive_sources = {"reviewed", *DOMAIN_POSITIVE_SOURCES}
    other_positives = [
        row
        for row in rows
        if row["positive"] == "1" and row["source"] not in known_positive_sources
    ]
    selected_rows = negatives + selected_web + other_positives
    selected_rows.extend(selected_domain * args.domain_repeat)
    rng.shuffle(selected_rows)

    output_list = args.output_list.resolve()
    output_yaml = args.output_yaml.resolve()
    output_summary = args.output_summary.resolve()
    for output in (output_list, output_yaml, output_summary):
        output.parent.mkdir(parents=True, exist_ok=True)
    paths = [image_path(data_dir, row) for row in selected_rows]
    output_list.write_text(
        "\n".join(str(path) for path in paths) + "\n", encoding="utf-8"
    )

    base_yaml_path = data_dir / "data.yaml"
    data_yaml = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
    data_yaml["path"] = str(data_dir)
    data_yaml["train"] = str(output_list)
    output_yaml.write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    unique_selected = {str(image_path(data_dir, row)) for row in selected_rows}
    summary = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "web_positive_limit": args.web_positive_limit,
        "domain_group_cap": args.domain_group_cap,
        "domain_repeat": args.domain_repeat,
        "original_train_images": len(rows),
        "train_occurrences": len(selected_rows),
        "unique_train_images": len(unique_selected),
        "negative_occurrences": len(negatives),
        "web_positive_occurrences": len(selected_web),
        "domain_positive_unique": len(selected_domain),
        "domain_positive_occurrences": len(selected_domain) * args.domain_repeat,
        "other_positive_occurrences": len(other_positives),
        "occurrences_by_source": dict(Counter(row["source"] for row in selected_rows)),
        "unique_by_source": dict(
            Counter(
                row["source"]
                for row in {str(image_path(data_dir, row)): row for row in selected_rows}.values()
            )
        ),
    }
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
