import numpy as np

from .scale_int12 import ScaleInt12


def _output_scale(value: float) -> ScaleInt12:
    # A nonzero tensor must never receive a zero scale at the UINT4 floor.
    return ScaleInt12(max(float(value), 2.0**-15))


def symmetric_quantize(
    x: np.ndarray,
    scale: ScaleInt12 | float | None = None,
    type: np.dtype = np.int8,
) -> tuple[np.ndarray, ScaleInt12]:
    """Per-tensor quantization; symmetric INT8 codes are in [-127, 127].

    Static scales may retain wide intermediate fields for INT32 biases.
    Bias overflow raises instead of saturating.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Input must be non-empty and finite")
    dtype = np.dtype(type)
    if dtype not in (np.dtype(np.int8), np.dtype(np.int32)):
        raise ValueError("Quantized dtype must be int8 or int32")
    limit = np.iinfo(dtype).max
    if scale is None:
        scale = _output_scale(float(np.max(np.abs(x))) / limit)
    elif not isinstance(scale, ScaleInt12):
        scale = ScaleInt12(float(scale))
    if not np.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Scale must be finite and positive")
    codes = np.rint(x / float(scale))
    if dtype == np.dtype(np.int32):
        bounds = np.iinfo(np.int32)
        if np.any(codes < bounds.min) or np.any(codes > bounds.max):
            raise OverflowError("Bias codes exceed INT32; scale is too small")
    else:
        codes = np.clip(codes, -limit, limit)
    return codes.astype(dtype), scale


def symmetric_dequantize(x_q: np.ndarray, scale: ScaleInt12 | float) -> np.ndarray:
    """Convert codes and a possibly wide scale to float64."""
    return np.asarray(x_q, dtype=np.float64) * float(scale)


def requantize(accumulator: np.ndarray, scale: ScaleInt12) -> tuple[np.ndarray, ScaleInt12]:
    """S_Y = max(abs(Y_hat))*S_acc/127; round only at the INT8 output."""
    accumulator = np.asarray(accumulator)
    if accumulator.size == 0 or not np.issubdtype(accumulator.dtype, np.signedinteger):
        raise ValueError("Accumulator must be a non-empty signed integer tensor")
    if not isinstance(scale, ScaleInt12) or not np.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Accumulator scale must be a positive ScaleInt12")
    peak = max(abs(int(accumulator.min())), abs(int(accumulator.max())))
    output_scale = _output_scale(peak * float(scale) / 127)
    codes = np.rint(accumulator.astype(np.float64) * (float(scale) / float(output_scale)))
    return np.clip(codes, -127, 127).astype(np.int8), output_scale


def add_quantized(left, right, *, wide=False):
    """Align and add residuals; optionally defer requantization to LayerNorm."""
    left_q, left_scale = left
    right_q, right_scale = right
    if np.shape(left_q) != np.shape(right_q):
        raise ValueError("Residual tensors must have matching shapes")
    degree = max(left_scale.degree, right_scale.degree)
    left_factor = int(left_scale.multiplier) << (degree - left_scale.degree)
    right_factor = int(right_scale.multiplier) << (degree - right_scale.degree)
    accumulator = (
        np.asarray(left_q, dtype=np.int64) * left_factor
        + np.asarray(right_q, dtype=np.int64) * right_factor
    )
    # 1*2**(-degree) is the exact common unit; don't normalize it to zero.
    unit = ScaleInt12.from_parts(1, degree, normalize=False)
    return (accumulator, unit) if wide else requantize(accumulator, unit)
