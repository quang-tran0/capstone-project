---
tags: [minilm, fpga, pooling, cosine, rag]
---

# Chương 12 — Mean pooling và cosine

← [[11 - GELU và residual]] | Tiếp: [[13 - rsqrt và nghịch đảo]] →

Code: `model/numpy/layers/mean_pooling.py`, `model/numpy/utils/similarity.py`

Sau 12 encoder layer, mỗi token có một vector 384 chiều. Chương này gộp chúng thành **một
vector cho cả câu**, rồi so khớp hai câu bằng cosine để tính score RAG.

## 12.1. Mean pooling

### Công thức gốc

$$
\bar{h}_d = \frac{1}{c} \sum_{i:\,M_i = 1} h_{id}, \qquad c = \sum_i M_i
$$

Lấy trung bình các token thật ($M_i = 1$), bỏ qua token padding.

### Lúc chạy

1. **Cộng dồn số nguyên** các token có mask = 1: $\Sigma_d = \sum_i M_i [h_{id}]$ (Q.16).
2. **Nhân với nghịch đảo của số token.** $1/c$ là một `Scale` từ `reciprocal_fixed`
   ([[13 - rsqrt và nghịch đảo]]), không cần phép chia:

```python
inverse_count = reciprocal_fixed(counts, self.reciprocal_lut)
return inverse_count.rescale(summed, 16)          # Q.16
```

$$
[\bar{h}_d] = \operatorname{round\_shift}(\Sigma_d \cdot y,\ s), \qquad \frac{1}{c} \approx y \cdot 2^{-s}
$$

Kết quả là **sentence embedding** Q.16, lưu trong thanh ghi 32 bit, shape $(B, 384)$. Đây là dữ
liệu FPGA gửi về phần mềm.

## 12.2. Cosine (score RAG)

### Công thức gốc

$$
\operatorname{score}(a, b) = \frac{a \cdot b}{\lVert a \rVert\, \lVert b \rVert} \in [-1, 1]
$$

Score càng gần 1 thì hai câu càng giống nghĩa. Score âm chỉ có nghĩa là không liên quan.

### Bước 1 — Thu nhỏ mỗi vector về 16 bit

Giống bước 3 của [[10 - LayerNorm]]: dịch cả vector sao cho phần tử lớn nhất vừa 16 bit có dấu:

$$
\hat{a} = a \gg k_a, \qquad k_a = \max\!\big(\operatorname{leading\_one}(\max |a|) + 1 - 16,\ 0\big)
$$

Scale của mỗi vector **bị triệt tiêu** trong cosine (tử và mẫu cùng chia), nên phép dịch không làm
đổi kết quả.

### Bước 2 — Tích vô hướng và chuẩn

$$
\mathrm{dot} = \sum_d \hat{a}_d \hat{b}_d, \qquad
\frac{1}{\lVert \hat{a} \rVert} \approx y_a \cdot 2^{-s_a}, \qquad
\frac{1}{\lVert \hat{b} \rVert} \approx y_b \cdot 2^{-s_b}
$$

Hai nghịch đảo căn được tính bằng `rsqrt_fixed` từ $\sum \hat a^2$ và $\sum \hat b^2$.

### Bước 3 — Chia lần lượt

Chia cho $\lVert \hat{a} \rVert$ trước để kết quả trung gian không vượt 64 bit, rồi mới chia cho
$\lVert \hat{b} \rVert$:

```python
partial = inverse_a.rescale(dot, frac=15)
score = inverse_b.rescale(partial, frac=15)       # Q.15, thanh ghi 17 bit
```

Kết quả ở **Q.15**: score $\in [-1, 1]$ nên cần 17 bit có dấu.

### Gợi ý cho hệ thống RAG

Vector tài liệu có thể được **chuẩn hoá sẵn** khi tạo chỉ mục (chia cho độ dài một lần). Khi
đó lúc truy vấn chỉ còn tích vô hướng và một lần rsqrt cho câu hỏi.

## 12.3. Ví dụ kết quả

Query "Hợp đồng bị vô hiệu khi nào?", vài passage:

| Passage | PyTorch | NumPy | Sai số |
|---|---|---|---|
| Thời hiệu khởi kiện … tranh chấp hợp đồng … | 0.55283 | 0.55554 | 0.0027 |
| Giao dịch dân sự do người chưa thành niên … vô hiệu … | 0.48071 | 0.46857 | 0.0121 |
| Theo Bộ luật Dân sự, hợp đồng là sự thỏa thuận … | 0.46739 | 0.46552 | 0.0019 |
| Python là ngôn ngữ lập trình phổ biến … | −0.21374 | −0.21906 | 0.0053 |

Score âm như −0.21 là bình thường với passage không liên quan.

---

← [[11 - GELU và residual]] | Tiếp: [[13 - rsqrt và nghịch đảo]] →
