import numpy as np

if __package__:
    from .fixed import check_bits, signed_bits
else:
    from fixed import check_bits, signed_bits


# Cả ba bảng được đưa về cùng định dạng Q.24 để cộng trực tiếp.
EMBED_FRAC_BITS = 24
EMBEDDING_NAMES = ("word_embeddings", "position_embeddings", "token_type_embeddings")


def prepare_embedding(tensors, name, fraction_bits):
    """Chạy trên máy tính để chuẩn bị dữ liệu cho FPGA."""
    prefix = f"embeddings.{name}.weight"

    weight = tensors[f"{prefix}_quantized"]
    scale = float(tensors[f"{prefix}_scale"].item())
    zero_point = int(tensors[f"{prefix}_zero_point"].item())

    # Hệ số nguyên biểu diễn scale với fraction_bits bit phần lẻ.
    multiplier = round(scale * (1 << fraction_bits))

    return weight, zero_point, multiplier


def prepare_embeddings(tensors, fraction_bits=EMBED_FRAC_BITS):
    """Chuẩn bị cả ba bảng và tính độ rộng bit của tổng embedding."""
    tables = {
        name: prepare_embedding(tensors, name, fraction_bits)
        for name in EMBEDDING_NAMES
    }

    # Cận trên |word + position + token_type| để chọn độ rộng thanh ghi.
    bound = sum(
        max(zero_point, 255 - zero_point) * multiplier
        for _, zero_point, multiplier in tables.values()
    )

    return tables, signed_bits(bound)


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


def embeddings_fixed(input_ids, token_type_ids, tables, sum_bits):
    """word + position + token_type, kết quả Q.EMBED_FRAC_BITS trong sum_bits bit."""
    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)

    if input_ids.shape != token_type_ids.shape:
        raise ValueError("input_ids và token_type_ids phải cùng shape")

    # Position ID là bộ đếm 0..seq_len-1 trong từng câu.
    position_ids = np.broadcast_to(np.arange(input_ids.shape[-1]), input_ids.shape)

    x = (
        lookup_embedding_fixed(input_ids, *tables["word_embeddings"])
        + lookup_embedding_fixed(position_ids, *tables["position_embeddings"])
        + lookup_embedding_fixed(token_type_ids, *tables["token_type_embeddings"])
    )

    return check_bits(x, sum_bits, "tổng embedding")
