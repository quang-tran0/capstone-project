import numpy as np


def prepare_embedding(tensors, name, fraction_bits):
    """Chạy trên máy tính để chuẩn bị dữ liệu cho FPGA."""
    prefix = f"embeddings.{name}.weight"

    weight = tensors[f"{prefix}_quantized"]
    scale = float(tensors[f"{prefix}_scale"].item())
    zero_point = int(tensors[f"{prefix}_zero_point"].item())

    # Hệ số nguyên biểu diễn scale với fraction_bits bit phần lẻ.
    multiplier = round(scale * (1 << fraction_bits))

    return weight, zero_point, multiplier


def lookup_embedding_fixed(ids, weight, zero_point, multiplier):
    """Đường tính số nguyên dùng để mô phỏng FPGA."""
    ids = np.asarray(ids)

    if ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("ids phải là mảng số nguyên không rỗng")

    if ids.min() < 0 or ids.max() >= weight.shape[0]:
        raise ValueError("ID nằm ngoài bảng embedding")

    # Đọc bảng: UINT8.
    quantized = weight[ids]

    # Mở rộng sang signed trước khi trừ.
    centered = quantized.astype(np.int32) - np.int32(zero_point)

    # Kết quả INT32 có fraction_bits bit phần lẻ.
    fixed = centered * np.int32(multiplier)

    return fixed

def word_embedding(input_ids, tensors):
    prefix = "embeddings.word_embeddings.weight"

    weight = tensors[f"{prefix}_quantized"]
    scale = np.float32(tensors[f"{prefix}_scale"].item())
    zero_point = np.int16(tensors[f"{prefix}_zero_point"].item())

    input_ids = np.asarray(input_ids, dtype=np.int64)

    if input_ids.size == 0:
        raise ValueError("input_ids không được rỗng")

    if input_ids.min() < 0 or input_ids.max() >= weight.shape[0]:
        raise ValueError("token ID nằm ngoài bảng word embedding")

    # 1. Đọc vector UINT8 ứng với từng token ID.
    quantized = weight[input_ids]


    # 2. Mở rộng sang INT16 trước khi trừ, tránh tràn UINT8.
    print(zero_point)
    centered = quantized.astype(np.int16) - zero_point
    # print(centered)

    # 3. Khôi phục giá trị FLOAT32 để đối chiếu với PyTorch.
    print(scale)
    embedding = centered.astype(np.float32) * scale
    print(embedding)

    return quantized, centered, embedding
