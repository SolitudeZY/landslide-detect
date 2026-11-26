#!/usr/bin/env python3
"""Train a compact Ultralytics YOLO detector for landslide detection."""

from __future__ import annotations

import argparse
import json
from numbers import Number
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    local_model = project_dir / "yolov8/ckpt/yolov8n.pt"
    default_model = str(local_model) if local_model.is_file() else "yolov8n.pt"
    parser = argparse.ArgumentParser(description="训练单类别滑坡 YOLO 检测模型。")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "dataset/landslide_yolo_screened_2026-08-21/data.yaml",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="RK3568 优先使用 n 级模型；也可传本地 .pt 继续训练。",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0", help="如 0、0,1、cpu。")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=script_dir / "runs/train")
    parser.add_argument("--name", default="landslide_yolov8n_screened_2026-08-21")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--fraction", type=float, default=1.0, help="训练集使用比例，冒烟测试可设 0.01。")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--evaluate-real-test",
        type=Path,
        help="可选的真实 test data YAML；用于和含合成图的完整 test 分开报告。",
    )
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("fraction 必须在 (0, 1] 范围内。")
    return args


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_path}")
    cache: bool | str = False if args.cache == "false" else args.cache
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        translate=args.translate,
        scale=args.scale,
        close_mosaic=args.close_mosaic,
        save_period=args.save_period,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        pretrained=True,
        cos_lr=True,
        cache=cache,
        resume=args.resume,
        plots=True,
        val=True,
        fraction=args.fraction,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights/best.pt"
    metadata = {
        "data": str(data_path),
        "model": args.model,
        "best": str(best.resolve()),
        "imgsz": args.imgsz,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "mosaic": args.mosaic,
        "translate": args.translate,
        "scale": args.scale,
    }
    if args.export_onnx:
        from export_utils import export_static_onnx

        exported = export_static_onnx(
            YOLO(str(best)),
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
        )
        metadata["onnx"] = str(exported)
    if args.evaluate_test:
        test_metrics = YOLO(str(best)).val(
            data=str(data_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project.resolve()),
            name=f"{args.name}_test",
        )
        metadata["test_metrics"] = {
            key: float(value)
            for key, value in test_metrics.results_dict.items()
            if isinstance(value, Number)
        }
    if args.evaluate_real_test:
        real_test_data = args.evaluate_real_test.resolve()
        if not real_test_data.is_file():
            raise FileNotFoundError(f"真实 test 数据配置不存在: {real_test_data}")
        real_test_metrics = YOLO(str(best)).val(
            data=str(real_test_data),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project.resolve()),
            name=f"{args.name}_real_test",
        )
        metadata["real_test_data"] = str(real_test_data)
        metadata["real_test_metrics"] = {
            key: float(value)
            for key, value in real_test_metrics.results_dict.items()
            if isinstance(value, Number)
        }
    (save_dir / "landslide_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
