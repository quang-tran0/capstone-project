import numpy as np
from scipy.special import erf


def dynamic_quantize(x, trace=None, name="Dynamic quantization"):
    x = np.asarray(x, dtype=np.float32)
    lower = np.minimum(x.min(), np.float32(0.0))
    upper = np.maximum(x.max(), np.float32(0.0))
    scale = np.float32((upper - lower) / np.float32(255.0))
    if scale == 0:
        scale = np.float32(1.0)

    zero_point = np.uint8(np.rint(np.clip(-lower / scale, 0, 255)))
    quantized = np.clip(np.rint(x / scale) + zero_point, 0, 255).astype(np.uint8)
    if trace:
        trace.operation(
            name,
            "scale=(max(max(x),0)-min(min(x),0))/255; q=clip(round(x/scale)+zp,0,255)",
        )
        trace.tensor("input", x)
        trace.scalar("lower", lower)
        trace.scalar("upper", upper)
        trace.scalar("scale", scale)
        trace.scalar("zero_point", zero_point)
        trace.tensor("quantized", quantized)
    return quantized, scale, zero_point


def quantized_linear(
    x,
    weight,
    weight_scale,
    weight_zero_point,
    bias,
    trace=None,
    name="Quantized linear",
):
    quantized, input_scale, input_zero_point = dynamic_quantize(
        x, trace, f"{name}: dynamic input quantization"
    )
    centered_input = quantized.astype(np.int32) - np.int32(input_zero_point)
    centered_weight = weight.astype(np.int32) - np.int32(weight_zero_point)
    accumulator = np.matmul(centered_input, centered_weight)
    output = accumulator.astype(np.float32) * (
        input_scale * np.float32(weight_scale)
    )
    result = (output + bias).astype(np.float32, copy=False)
    if trace:
        trace.operation(
            name,
            "y=((q_x-zp_x) @ (q_w-zp_w)) * scale_x * scale_w + bias",
        )
        trace.tensor("weight", weight, parameter=True)
        trace.scalar("weight_scale", weight_scale)
        trace.scalar("weight_zero_point", weight_zero_point)
        trace.tensor("bias", bias, parameter=True)
        trace.tensor("centered_input", centered_input)
        trace.tensor("centered_weight", centered_weight, parameter=True)
        trace.tensor("int32_accumulator", accumulator)
        trace.tensor("rescaled", output)
        trace.tensor("output", result)
    return result


def layer_norm(x, weight, bias, eps, trace=None, name="Layer normalization"):
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    denominator = np.sqrt(variance + np.float32(eps), dtype=np.float32)
    normalized = centered / denominator
    output = (normalized * weight + bias).astype(np.float32, copy=False)
    if trace:
        trace.operation(
            name,
            "y=((x-mean(x))/sqrt(mean((x-mean(x))^2)+eps))*weight+bias",
        )
        trace.tensor("input", x)
        trace.tensor("weight", weight, parameter=True)
        trace.tensor("bias", bias, parameter=True)
        trace.scalar("epsilon", eps)
        trace.tensor("mean", mean)
        trace.tensor("centered", centered)
        trace.tensor("squared_centered", centered * centered)
        trace.tensor("variance", variance)
        trace.tensor("denominator", denominator)
        trace.tensor("normalized", normalized)
        trace.tensor("output", output)
    return output


def softmax(x, axis=-1, trace=None, name="Softmax"):
    x = np.asarray(x, dtype=np.float32)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponential = np.exp(shifted, dtype=np.float32)
    denominator = np.sum(exponential, axis=axis, keepdims=True, dtype=np.float32)
    output = (exponential / denominator).astype(np.float32, copy=False)
    if trace:
        trace.operation(name, "y=exp(x-max(x))/sum(exp(x-max(x)))")
        trace.scalar("axis", axis)
        trace.tensor("input", x)
        trace.tensor("maximum", np.max(x, axis=axis, keepdims=True))
        trace.tensor("shifted", shifted)
        trace.tensor("exponential", exponential)
        trace.tensor("denominator", denominator)
        trace.tensor("output", output)
    return output


def gelu(x, trace=None, name="GELU"):
    x = np.asarray(x, dtype=np.float32)
    scaled = x / np.float32(np.sqrt(2.0))
    erf_value = erf(scaled)
    output = np.float32(0.5) * x * (np.float32(1.0) + erf_value)
    output = output.astype(np.float32, copy=False)
    if trace:
        trace.operation(name, "y=0.5*x*(1+erf(x/sqrt(2)))")
        trace.tensor("input", x)
        trace.tensor("x_div_sqrt_2", scaled)
        trace.tensor("erf", erf_value)
        trace.tensor("one_plus_erf", np.float32(1.0) + erf_value)
        trace.tensor("output", output)
    return output
