---
tags: [minilm, fpga, kiem-thu]
---

# Chương 15 — Kiểm chứng và độ chính xác

← [[14 - Độ rộng bit và đưa lên FPGA]] | Tiếp: [[16 - Thuật ngữ]] →

## 15.1. Ba lớp kiểm chứng

| Lớp | Kiểm tra gì | Công cụ |
|---|---|---|
| Mô hình float tham chiếu | Model PyTorch tự viết có đúng như ONNX gốc không | So với chính đồ thị ONNX |
| Mô hình số nguyên | NumPy sai bao nhiêu so với PyTorch float32 | `model/main.py` → `report.md` |
| Từng khối | Mỗi hàm số nguyên đúng công thức, mỗi kiểu dữ liệu đúng quy tắc | `test_numpy_model.py` |

## 15.2. Mô hình float tham chiếu

`model/pytorch/main.py` dựng lại MiniLM bằng PyTorch, nạp weight từ `model.onnx`. Để chắc nó đúng,
output được so với chính đồ thị ONNX (chạy bằng `onnx.reference.ReferenceEvaluator`):

| So sánh | Sai số tuyệt đối lớn nhất |
|---|---|
| `last_hidden_state` | $1.4 \times 10^{-6}$ |
| Sentence embedding | $3.7 \times 10^{-7}$ |
| Một câu chạy riêng vs nằm trong batch có padding | $3.6 \times 10^{-7}$ |

Các sai số này chỉ ở mức làm tròn float32. Cấu trúc cũng khớp: GELU dùng `erf`, LayerNorm
$\varepsilon = 10^{-12}$ đặt sau residual, position ID bắt đầu từ 0.

## 15.3. So sánh số nguyên với float: `report.md`

```bash
python model/main.py
```

Lệnh hỏi câu truy vấn và các đoạn tài liệu (Enter để dùng mặc định: 20 đoạn, 5 đoạn về luật dân
sự), rồi ghi `model/report.md` gồm:

1. **Bảng tóm tắt:** thứ hạng giống nhau không, sai số score, cosine embedding nhỏ nhất.
2. **Kết quả query** của từng bản: xếp hạng passage theo cosine.
3. **Sai số score RAG:** điểm hai bản, sai số, hạng; hạng khác nhau được in đậm.
4. **Sai số sentence embedding** của từng câu.
5. **Sai số theo layer:** bảng tóm tắt 12 layer, rồi bảng chi tiết từng bước trong mỗi layer
   (Q, K, V, attention, dense, LayerNorm, GELU…).

### Công thức sai số

$$
e_d = 100 \cdot \frac{\lvert \hat{y}_d - y_d \rvert}{\max(y) - \min(y)}\ (\%)
$$

$\hat y$ là kết quả NumPy (đổi sang số thực), $y$ là PyTorch. Chia cho **khoảng giá trị** của cả
tensor thay vì cho từng $y_d$: khi $y_d$ gần 0, chia cho chính nó sẽ ra phần trăm rất lớn dù sai
số thật nhỏ (đã từng ra 233% với score 0.004 lệch 0.009).

## 15.4. Kết quả

Query "Hợp đồng bị vô hiệu khi nào?", 20 đoạn mặc định:

| Đại lượng | INT8 một scale cho cả tensor (thiết kế cũ) | Hiện tại |
|---|---|---|
| Cosine sentence embedding | 0.847 – 0.914 | **0.9985 – 0.9997** |
| Sai số TB / lớn nhất của embedding | 7.1% / 25.6% | **0.33–0.78% / 1.3–3.7%** |
| Sai số score RAG lớn nhất | 0.087 | **0.012** |

**Thứ hạng:** top 3 giống hệt PyTorch (đều là các đoạn luật dân sự). Có 2 cặp passage đổi chỗ, cả
hai đều có score PyTorch rất sát nhau (chênh 0.008 và 0.0009).

**Sai số qua các layer:** tăng dần từ layer 0 đến layer 11, lớn nhất ở FFN của các layer cuối
(có chỗ max error khoảng 9%). Sau mỗi LayerNorm, sai số bị kéo xuống lại khoảng 1–1.7%.

| Layer (output) | Sai số TB | Sai số lớn nhất | Cosine |
|---|---|---|---|
| embeddings | 0.065% | 0.36% | 0.99984 |
| layer 0 | 0.060% | 0.35% | 0.99985 |
| layer 5 | 0.18% | 1.05% | 0.99937 |
| layer 11 | 0.28% | 3.51% | 0.99871 |
| pooling | 0.49% | 2.10% | 0.99935 |

## 15.5. Kiểm thử từng khối

```bash
python -m unittest model.test_numpy_model -v
```

24 test:

| Nhóm | Kiểm tra |
|---|---|
| `FixedTest` | Quy tắc độ rộng, trộn có dấu/không dấu, ba chế độ tràn, theo dõi `frac`, làm tròn khi dịch, LUT, bộ chọn |
| `ScaleTest` | Scale cấu hình được, lượng tử hoá với định dạng tuỳ chọn, `from_peak` so với float, nhân và rescale, `pack` |
| `IntLayerNormTest` | So với LayerNorm float, scale input bị triệt tiêu, hàng hằng số ra $\beta$ |
| `ArithmeticTest` | rsqrt, nghịch đảo, lượng tử hoá, softmax, GELU, Linear, pooling, cosine so với float |
| `EndToEndTest` | Cả mô hình so với PyTorch (cần `model.onnx`) |

## 15.6. Khớp từng bit khi đổi cài đặt

Khi chuyển mô hình từ mảng `int64` sang các kiểu `Fixed`/`Scale`, toàn bộ output và trace trung
gian (273 tensor, gồm cả batch có padding) được lưu trước rồi so lại sau: **khớp hoàn toàn từng
bit**. Cách này nên dùng cho mọi lần sửa mô hình sau này mà không định đổi kết quả.

---

← [[14 - Độ rộng bit và đưa lên FPGA]] | Tiếp: [[16 - Thuật ngữ]] →
