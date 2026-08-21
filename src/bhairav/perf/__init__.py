"""Performance optimization package (Phase 15).

ONNX model export + quantization, batched multi-camera inference,
TensorRT acceleration, and inference profiling.
"""
from .onnx_export import export_yolo_to_onnx, quantize_onnx_model
from .batched import BatchedInferenceEngine
from .profiler import InferenceProfiler

__all__ = [
    "export_yolo_to_onnx",
    "quantize_onnx_model",
    "BatchedInferenceEngine",
    "InferenceProfiler",
]
