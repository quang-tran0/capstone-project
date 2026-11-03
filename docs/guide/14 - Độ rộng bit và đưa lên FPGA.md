---
tags: [minilm, fpga, rtl, kv260]
---

# Chương 14 — Độ rộng bit và đưa lên FPGA

← [[13 - rsqrt và nghịch đảo]] | Tiếp: [[15 - Kiểm chứng và độ chính xác]] →

Chương này tổng hợp mọi thanh ghi, bảng tra và bộ nhớ, rồi đánh giá việc đưa luồng tính lên
FPGA KV260 (chip `xck26-sfvc784-2LV-c`).

> [!warning] Đây là đánh giá trên giấy
> Nội dung dựa trên mô hình NumPy. Chưa có RTL, chưa chạy Vivado, nên chưa có số timing hay
> tài nguyên. Mỗi module RTL sau này phải chạy timing và utilization sau place-and-route mới
> được coi là đạt.

## 14.1. Kết luận chung

Luồng tính **làm được trên FPGA**: toàn bộ chỉ có nhân, cộng, dịch bit, so sánh và tra bảng,
đúng những phép FPGA làm tốt. Nhưng có ba điểm cần xử lý trước khi viết RTL (mục 14.5).

## 14.2. Các thanh ghi đã khai báo

Mỗi dòng dưới đây là một chỗ gọi `resize` trong mô hình ([[03 - Kiểu dữ liệu phần cứng Fixed#3.5. Khai báo thanh ghi bằng resize]]).
Nếu dữ liệu không vừa, mô hình báo lỗi ngay.

| Thanh ghi | Độ rộng | Chương |
|---|---|---|
| Đầu ra lookup embedding | 32 bit, Q.16 | [[08 - Embedding]] |
| Mã INT8 sau lượng tử hoá | 8 bit | [[06 - Lượng tử hoá]] |
| Accumulator `Linear` | 32 bit | [[07 - Linear]] |
| Đầu ra `Linear` | 32 bit, Q.16 | [[07 - Linear]] |
| Accumulator điểm attention | 20 bit | [[09 - Self-attention và softmax]] |
| Mã xác suất softmax | 8 bit | [[09 - Self-attention và softmax]] |
| Accumulator context | 22 bit | [[09 - Self-attention và softmax]] |
| Đầu ra context | 32 bit, Q.16 | [[09 - Self-attention và softmax]] |
| LayerNorm: $e$ sau khi thu nhỏ | 18 bit | [[10 - LayerNorm]] |
| LayerNorm: $\sum e^2$ | 48 bit | [[10 - LayerNorm]] |
| LayerNorm: đầu ra | 32 bit, Q.16 | [[10 - LayerNorm]] |
| GELU: đầu ra | 32 bit, Q.16 | [[11 - GELU và residual]] |
| Mantissa rsqrt / nghịch đảo | 17 bit không dấu | [[13 - rsqrt và nghịch đảo]] |
| Pooling: đầu ra | 32 bit, Q.16 | [[12 - Mean pooling và cosine]] |
| Cosine: vector thu nhỏ / score | 16 bit / 17 bit (Q.15) | [[12 - Mean pooling và cosine]] |

## 14.3. Hằng số và bảng tra

| Tên | Kích thước | Định dạng |
|---|---|---|
| $R = \lceil 2^{24}/127 \rceil$ | 1 hằng: 132 105 | chia cho 127 khi chọn scale |
| $1/\sqrt{32}$ | `Scale.const(5793, 15)` | điểm attention |
| $\log_2 e$ | 1 hằng: 47 274 | Q.15 |
| ROM $1/m$ (lượng tử hoá) | 128 ô | 23 bit không dấu |
| EXP2 (softmax) | 257 ô | 16 bit, Q.15 |
| PHI (GELU) | 1024 ô | 16 bit, Q.15 |
| RSQRT | 768 ô | 17 bit, Q.16 |
| RECIP (nghịch đảo) | 256 ô | 17 bit, Q.16 |

Tất cả đều nhỏ, vừa LUTRAM hoặc một BRAM mỗi bảng.

## 14.4. Những phần map tốt sang DSP48E2

DSP48E2 nhân $27 \times 18$ bit và cộng dồn 48 bit.

| Phép tính | Độ rộng | Ghi chú |
|---|---|---|
| MAC INT8 × INT8 | acc $\le 25$ bit | Có thể ghép 2 MAC INT8 vào một DSP |
| Rescale `acc · m_x m_w` | $25 \times 14$ bit → 39 bit | Vừa 1 DSP |
| `m_x · m_w` | $7 \times 7$ bit | LUT hoặc một DSP nhỏ |
| LayerNorm $e^2$, cộng dồn | $18 \times 18$, 48 bit | Đúng kích thước DSP |
| LayerNorm `e·m`, `norm·G` | $18 \times 17$, $18 \times 18$ | Vừa 1 DSP mỗi phép |
| Điểm attention, context | 20 / 22 bit | Nhỏ |

## 14.5. Những chỗ cần sửa trước khi viết RTL

### (1) Phép nhân quá rộng khi lượng tử hoá

Mỗi phần tử input của mỗi `Linear` được tính $x \cdot \mathrm{RECIP}[m]$: $x$ 32 bit nhân 23 bit
ra **khoảng 55 bit**, cần 3–4 DSP cho mỗi phần tử. Phép này chạy cho mọi phần tử (384 hoặc 1536
mỗi token) nên rất tốn.

**Cách khắc phục:** mã ra chỉ có 8 bit, nên có thể dịch $x$ xuống trước khi nhân (đã biết $n$ thì
biết bỏ được bao nhiêu bit thấp), sao cho cả hai thừa số vừa DSP. Phải **sửa mô hình NumPy
trước** để giữ khớp từng bit với RTL.

Tương tự, $P \cdot R$ khi chọn scale cần khoảng 57 bit, nhưng mỗi token chỉ một lần, có thể làm
tuần tự.

### (2) Các phép cần quét một hàng nhiều lượt

- Lượng tử hoá theo token cần **đỉnh của cả hàng** trước khi tính mã đầu tiên.
- Softmax cần max và tổng của hàng.
- LayerNorm cần tổng, đỉnh và tổng bình phương.

Mỗi chỗ phải đệm cả hàng: $384 \times 32$ bit = 1.5 KB, hoặc $1536 \times 32$ bit = 6 KB cho đầu
vào dense2. BRAM chứa thoải mái; cái giá là thêm độ trễ, hoặc phải thiết kế pipeline hai pha.

### (3) Bộ nhớ: trọng số không vừa on-chip

| Dữ liệu | Kích thước (INT8) |
|---|---|
| Bảng word embedding | ≈ 96 MB |
| Weight encoder, mỗi layer | ≈ 1.77 MB |
| Weight encoder, 12 layer | ≈ 21 MB |
| Bộ nhớ on-chip XCK26 (144 BRAM36 + 64 URAM) | ≈ 2.9 MB |

Weight phải được **stream từ DDR4** qua AXI theo từng layer hoặc từng tile. Một layer (1.77 MB)
có thể nằm vừa trong URAM nhưng không đủ chỗ để double-buffer. Embedding nên để phần mềm tra.

### (4) Bộ dịch thay đổi theo phần tử

Lượng dịch $n_x + n_w - 16$ thay đổi theo từng kênh, nên mỗi làn output cần một **barrel shifter**.
Với vài trăm làn song song, phần này tốn đáng kể LUT. Vì $n_w$ chỉ nằm trong một khoảng hẹp, nếu
chuẩn hoá offline cho mọi kênh dùng chung $n_w$ thì lượng dịch không đổi trong cả hàng và barrel
shifter gần như biến mất. Cũng phải sửa trong NumPy trước.

## 14.6. Ước lượng thô về tốc độ

Câu 64 token qua 12 layer cần khoảng **2.7 tỉ phép MAC**. Một systolic array 1024 MAC (khoảng 512
DSP nếu ghép 2 MAC mỗi DSP) ở 200 MHz mất **khoảng 13 ms** cho phần tính. Đọc 21 MB weight từ DDR
thêm vài ms. KV260 có 1248 DSP nên con số này nằm trong khả năng, nhưng mới là ước lượng.

## 14.7. Thứ tự làm việc đề xuất

1. Sửa mô hình NumPy: thu hẹp hai phép nhân rộng, chuẩn hoá $n_w$ nếu muốn bỏ barrel shifter.
2. Chạy lại so sánh với PyTorch ([[15 - Kiểm chứng và độ chính xác]]) để chắc độ chính xác không
   giảm.
3. Viết RTL từng khối, dùng mô hình NumPy làm đáp án cho testbench (khớp từng bit).
4. Mỗi khối RTL chạy timing (setup, hold, pulse width) và utilization sau place-and-route trên
   `xck26-sfvc784-2LV-c`, tại clock mục tiêu.

---

← [[13 - rsqrt và nghịch đảo]] | Tiếp: [[15 - Kiểm chứng và độ chính xác]] →
