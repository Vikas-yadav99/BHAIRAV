"""ONNX model export + quantization (Phase 15).

Converts ultralytics YOLO models to ONNX for faster cross-platform
inference, then optionally quantises to INT8 for edge deployment.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_yolo_to_onnx(
    model_name: str = "yolov8n.pt",
    output_path: str | Path | None = None,
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
) -> Path:
    """Export a YOLO model to ONNX format via ultralytics.

    Parameters
    ----------
    model_name : str
        YOLO model name or path (e.g. ``yolov8n.pt``).
    output_path : str | Path | None
        Where to save the ONNX file.  Defaults to ``<model_stem>.onnx``
        in the current directory.
    imgsz : int
        Export input resolution (square).
    half : bool
        Export in FP16 half-precision.
    simplify : bool
        Run ``onnx-simplifier`` on the exported graph.

    Returns
    -------
    Path
        The path to the exported ONNX file.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "ultralytics is required for ONNX export. "
            "Install with: pip install ultralytics"
        )

    model = YOLO(model_name)
    out_dir = Path(output_path).parent if output_path else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=17,
    )
    elapsed = time.perf_counter() - t0

    # ultralytics returns the exported path
    onnx_path = Path(result) if isinstance(result, (str, Path)) else Path(str(result))
    if output_path and Path(output_path) != onnx_path:
        import shutil
        shutil.copy2(str(onnx_path), str(output_path))
        onnx_path = Path(output_path)

    logger.info("Exported %s -> %s (%.1fs)", model_name, onnx_path, elapsed)
    return onnx_path


def quantize_onnx_model(
    model_path: str | Path,
    output_path: str | Path | None = None,
    calibration_data: np.ndarray | None = None,
    method: str = "dynamic",
) -> Path:
    """Quantize an ONNX model to INT8 for faster edge inference.

    Parameters
    ----------
    model_path : str | Path
        Input ONNX model.
    output_path : str | Path | None
        Output path.  Defaults to ``<stem>_int8.onnx``.
    calibration_data : np.ndarray | None
        Calibration data for static quantisation (N, C, H, W) float32.
        If None and method is "dynamic", calibration is not needed.
    method : str
        ``"dynamic"`` (no calibration needed, quantises at runtime) or
        ``"static"`` (uses calibration data for better accuracy).

    Returns
    -------
    Path
        Path to the quantized ONNX model.
    """
    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError:
        raise ImportError(
            "onnxruntime[quantization] is required. "
            "Install with: pip install onnxruntime"
        )

    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.with_name(model_path.stem + "_int8.onnx")
    output_path = Path(output_path)

    if method == "dynamic":
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    elif method == "static":
        if calibration_data is None:
            raise ValueError("calibration_data is required for static quantisation")
        quantize_static(
            model_input=str(model_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_data,
        )
    else:
        raise ValueError(f"Unknown quantization method: {method!r}")

    logger.info("Quantized %s -> %s (method=%s)", model_path.name, output_path, method)
    return output_path


def get_optimal_provider() -> str:
    """Detect the best ONNX Runtime execution provider available.

    Priority: CUDA > TensorRT > CoreML > DirectML > CPU.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        available = ort.get_available_providers()
        for prov in ("CUDAExecutionProvider", "TensorrtExecutionProvider",
                      "CoreMLExecutionProvider", "DmlExecutionProvider"):
            if prov in available:
                return prov
    except ImportError:
        pass
    return "CPUExecutionProvider"
