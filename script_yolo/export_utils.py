"""Shared static ONNX export helpers."""

from __future__ import annotations

from pathlib import Path

import onnx
from ultralytics import YOLO


def export_static_onnx(
    model: YOLO,
    imgsz: int,
    opset: int,
    device: str,
    run_simplifier: bool = True,
) -> Path:
    output = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=opset,
            nms=False,
            device=device,
        )
    ).resolve()
    if run_simplifier:
        from onnxsim import simplify

        graph = onnx.load(str(output))
        simplified_graph, valid = simplify(graph)
        if not valid:
            raise RuntimeError("onnx-simplifier 校验失败")
        onnx.save(simplified_graph, str(output))
    return output
