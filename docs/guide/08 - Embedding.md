---
tags: [minilm, fpga, embedding]
---

# Chương 8 — Embedding

← [[07 - Linear]] | Tiếp: [[09 - Self-attention và softmax]] →

Code: `model/numpy/layers/bert_embedding.py`

## 8.1. Công thức gốc

Mỗi token được biểu diễn bằng tổng ba vector 384 chiều, rồi qua LayerNorm:

$$
x = \operatorname{LayerNorm}\big(E_{\text{word}}[\text{id}] + E_{\text{pos}}[\text{vị trí}] + E_{\text{type}}[\text{loại}]\big)
$$

| Bảng | Kích thước | Ý nghĩa |
|---|---|---|
| Word | $250\,037 \times 384$ | Mỗi token trong từ vựng một vector |
| Position | $512 \times 384$ | Mỗi vị trí trong câu một vector |
| Token type | $2 \times 384$ | Câu A / câu B (đồ án luôn dùng loại 0) |

## 8.2. Lưu trữ (offline)

Ba bảng được lượng tử hoá INT8, **mỗi hàng một scale** (xem [[06 - Lượng tử hoá#6.2. Weight (offline)]]).
Mỗi hàng là vector của một token, nên scale theo hàng giữ được độ chính xác cho cả token có giá
trị nhỏ.

## 8.3. Lúc chạy

Tra một hàng mã INT8 cùng scale của hàng đó, rồi đổi về Q.16:

$$
[e_k] = \operatorname{round\_shift}\!\left(q_k \cdot m,\ n - 16\right)
$$

```python
def _lookup(self, name, ids):
    table = self.tables[name]
    return table.scale[ids].rescale(table.codes[ids], FRAC_BITS).resize(32)
```

Phép nhân là 8 bit × 7 bit. Sau đó cộng ba vector (đều Q.16 nên cộng thẳng) và đưa qua LayerNorm:

$$
[x] = [e_{\text{word}}] + [e_{\text{pos}}] + [e_{\text{type}}]
$$

Position ID chỉ là bộ đếm $0, 1, \dots, L - 1$.

## 8.4. Trên phần cứng

Bảng word chiếm khoảng **96 MB** (INT8), không vừa bộ nhớ on-chip của KV260 (khoảng 2.9 MB). Vì
mỗi token chỉ cần đọc đúng một hàng, cách hợp lý là để phần mềm (ARM) tra bảng trong DDR rồi gửi
vector xuống, hoặc dùng DMA đọc hàng tương ứng. Hai bảng nhỏ (position, token type) có thể nằm
trong BRAM.

---

← [[07 - Linear]] | Tiếp: [[09 - Self-attention và softmax]] →
