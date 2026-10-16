import numpy as np


def symmetric_quantize(
    x: np.ndarray,
    scale: float | None = None
) -> tuple[np.ndarray, float]:
    """
    Symmetric INT8 Quantization (per-tensor).

    Args:
        x: Tensor FP32.
        scale:
            None  -> Dynamic Quantization.
            float -> Static Quantization.

    Returns:
        x_q: Tensor INT8.
        scale: Quantization scale.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Input must be non-empty and finite")

    if scale is None:
        max_abs = np.max(np.abs(x))
        scale = float(max_abs / 128.0)

        if scale == 0:
            scale = 1.0

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")

    x_q = np.rint(x / scale)
    x_q = np.clip(x_q, -128, 127)
    x_q = x_q.astype(np.int8)

    return x_q, scale


def symmetric_dequantize(
    x_q: np.ndarray,
    scale: float
) -> np.ndarray:
    """Convert INT8 back to FP32."""
    return x_q.astype(np.float32) * scale
