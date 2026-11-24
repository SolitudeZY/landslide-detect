#!/usr/bin/env python3
"""Create label-free night/street-light hard negatives from existing negatives."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_images <= 0 or args.copies <= 0:
        parser.error("max-images and copies must be positive")
    return args


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def night_variant(image: np.ndarray, rng: random.Random) -> np.ndarray:
    # Gamma and exposure model low-light cameras; channel gains model warm lamps.
    gamma = rng.uniform(1.35, 2.10)
    exposure = rng.uniform(0.55, 0.85)
    gains = np.array(
        [rng.uniform(0.55, 0.75), rng.uniform(0.78, 0.95), rng.uniform(0.98, 1.12)],
        dtype=np.float32,
    )  # BGR
    value = image.astype(np.float32) / 255.0
    value = np.power(value, gamma) * exposure
    value *= gains[None, None, :]
    result = np.clip(value * 255.0, 0, 255)

    if rng.random() < 0.75:
        noise_sigma = rng.uniform(1.5, 6.0)
        noise = np.random.default_rng(rng.randrange(1 << 32)).normal(
            0.0, noise_sigma, result.shape
        )
        result = np.clip(result + noise, 0, 255)
    if rng.random() < 0.35:
        result = cv2.GaussianBlur(result, (3, 3), 0)
    return result.astype(np.uint8)


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: min(args.max_images, len(paths))]
    written = 0
    skipped = 0
    for source in paths:
        source_id = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:12]
        try:
            image = read_image(source)
        except (OSError, ValueError):
            skipped += args.copies
            continue
        for index in range(args.copies):
            destination = args.output_dir / f"nightaug_{source_id}_v{index + 1}.jpg"
            if destination.exists() and not args.overwrite:
                continue
            variant = night_variant(image, rng)
            ok, encoded = cv2.imencode(
                ".jpg", variant, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if not ok:
                skipped += 1
                continue
            encoded.tofile(str(destination))
            written += 1
    print(
        f"sources={len(paths)} written={written} skipped={skipped} "
        f"output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
