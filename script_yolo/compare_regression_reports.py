#!/usr/bin/env python3
"""Compare old/new false-positive and miss sets at a fixed threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="对比新旧模型在固定误报和漏检集上的结果。")
    parser.add_argument("--old-negative", type=Path, required=True)
    parser.add_argument("--new-negative", type=Path, required=True)
    parser.add_argument("--old-positive", type=Path, required=True)
    parser.add_argument("--new-positive", type=Path, required=True)
    parser.add_argument("--threshold", default="0.35")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_negative = load(args.old_negative)
    new_negative = load(args.new_negative)
    old_positive = load(args.old_positive)
    new_positive = load(args.new_positive)
    threshold = args.threshold

    old_false_positives = set(
        old_negative["metrics"][threshold]["false_positive_files"]
    )
    new_false_positives = set(
        new_negative["metrics"][threshold]["false_positive_files"]
    )
    old_misses = set(old_positive["metrics"][threshold]["complete_miss_files"])
    new_misses = set(new_positive["metrics"][threshold]["complete_miss_files"])

    resolved_false_positives = old_false_positives - new_false_positives
    remaining_false_positives = old_false_positives & new_false_positives
    introduced_false_positives = new_false_positives - old_false_positives
    recovered_misses = old_misses - new_misses
    remaining_misses = old_misses & new_misses
    introduced_misses = new_misses - old_misses

    report = {
        "threshold": float(threshold),
        "negative_regression": {
            "old_false_positive_images": len(old_false_positives),
            "resolved_old_false_positives": len(resolved_false_positives),
            "remaining_old_false_positives": len(remaining_false_positives),
            "new_false_positive_images": len(introduced_false_positives),
            "resolved_files": sorted(resolved_false_positives),
            "remaining_files": sorted(remaining_false_positives),
            "new_files": sorted(introduced_false_positives),
        },
        "positive_regression": {
            "old_complete_miss_images": len(old_misses),
            "recovered_old_misses": len(recovered_misses),
            "remaining_old_misses": len(remaining_misses),
            "new_complete_miss_images": len(introduced_misses),
            "recovered_files": sorted(recovered_misses),
            "remaining_files": sorted(remaining_misses),
            "new_files": sorted(introduced_misses),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
