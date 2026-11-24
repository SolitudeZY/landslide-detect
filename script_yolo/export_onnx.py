#!/usr/bin/env python3
"""Export a static-shape ONNX model suitable for subsequent RKNN conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from export_utils import export_static_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="导出供 RKNN Toolkit2 转换的静态 ONNX。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    output = export_static_onnx(
        YOLO(str(args.model)),
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
        run_simplifier=not args.no_simplify,
    )
    print(output)


if __name__ == "__main__":
    main()
