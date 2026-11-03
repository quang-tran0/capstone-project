---
tags: [minilm, fpga, hwtypes, scale]
---

# Chương 4 — Scale $m \cdot 2^{-n}$

← [[03 - Kiểu dữ liệu phần cứng Fixed]] | Tiếp: [[05 - Cách lưu số]] →

Code: `model/numpy/hwtypes/scale.py`

## 4.1. Scale là gì

Khi lưu một tensor bằng mã INT8, ta cần một **hệ số tỉ lệ** (scale) $S$ để biết mỗi mã ứng với
giá trị thực nào:

$$
x \approx x_q \cdot S, \qquad x_q \in [-127, 127]
$$

Ví dụ: một hàng có giá trị lớn nhất 2.34 thì $S \approx 2.34 / 127 = 0.0184$. Mã 127 ứng với
2.34, mã 20 ứng với 0.37.

## 4.2. Vì sao dạng $m \cdot 2^{-n}$

Scale là số thực, mà FPGA không muốn làm việc với số thực. Đồ án viết mọi scale dưới dạng:

$$
S = m \cdot 2^{-n}, \qquad m, n \text{ là số nguyên không dấu}
$$

Khi đó nhân với $S$ chỉ cần **một bộ nhân** (với $m$) và **một bộ dịch** (dịch phải $n$ bit).
Không cần số thực, không cần phép chia.

Ví dụ: $S = 0.018555 \approx 76 \cdot 2^{-12}$ ($76 / 4096 = 0.018555$).

## 4.3. Định dạng mặc định: 7 + 5 bit

$$
m \in [1, 127]\ (7 \text{ bit}), \qquad n \in [0, 31]\ (5 \text{ bit})
$$

- **$m$ được chọn lớn nhất có thể** (với weight của MiniLM luôn nằm trong $[64, 127]$), nên luôn
  có đủ 7 bit có nghĩa.
- **$n$ cần 5 bit** vì scale của weight thường cỡ $10^{-3}$–$10^{-4}$, nhỏ hơn $2^{-15}$. Trên
  MiniLM, $n$ của weight nằm trong 11..18, của bảng word embedding trong 14..22. Với $n$ chỉ 4 bit
  ($\le 15$), các scale nhỏ sẽ có $m$ chỉ vài đơn vị và mất độ chính xác.
- Ghép lại, một scale chiếm **12 bit**.

## 4.4. `ScaleFormat`: cấu hình độ rộng

```python
ScaleFormat(m_bits=7, n_bits=5)
```

Độ rộng hai trường đổi được tuỳ ý, để thử các phương án phần cứng khác nhau:

| Thuộc tính / hàm | Ý nghĩa |
|---|---|
| `m_max`, `n_max` | $2^{m\_bits} - 1$ và $2^{n\_bits} - 1$ (mặc định 127 và 31) |
| `scale(m, n)` | Tạo `Scale`, kiểm tra hai trường vừa định dạng |
| `from_float(s)` | Offline: chọn $(m, n)$ cho số thực $s$ |
| `from_peak(peak, code_max)` | Lúc chạy, chỉ dùng số nguyên: chọn $(m, n) \approx \text{peak} / \text{code\_max}$ |

```python
fmt = ScaleFormat(m_bits=4, n_bits=3)    # m <= 15, n <= 7
q = quantize(x, scale_format=ScaleFormat(5, 4), code_bits=6)
```

Cách chọn $(m, n)$ được giải thích ở [[06 - Lượng tử hoá]].

## 4.5. `Scale`: các phép tính

`Scale(m, n)` gồm hai trường `Fixed` không dấu. Chúng có shape, nên một `Scale` có thể chứa
nhiều scale (mỗi token, mỗi cột một scale) và tự broadcast khi tính.

| Phép | Ý nghĩa |
|---|---|
| `S1 * S2` | $m = m_1 m_2$, $n = n_1 + n_2$; giữ đủ bit, không làm tròn |
| `S.rescale(value, frac)` | $\text{value} \cdot m \cdot 2^{-n}$, kết quả ở Q.`frac` |
| `S[idx]`, `S.swapaxes(...)` | Chọn hàng, đổi trục như mảng |
| `S.pack()` | Ghép $\{m, n\}$ thành một word (12 bit với mặc định) |
| `Scale.const(m, n)` | Hằng số, ví dụ $1/\sqrt{32} \approx$ `Scale.const(5793, 15)` |

### Nhân hai scale

$$
(m_1 2^{-n_1}) \cdot (m_2 2^{-n_2}) = (m_1 m_2) \cdot 2^{-(n_1 + n_2)}
$$

Trên phần cứng là một bộ nhân 7 × 7 bit (ra 14 bit) và một bộ cộng 5 bit.

### Rescale - nhân với scale và đổi định dạng

Đây là phép quan trọng nhất, dùng sau mọi phép nhân ma trận:

```python
def rescale(self, value, frac):
    return (value * self.m).round_shift(self.n + (value.frac - frac), frac=frac)
```

Nghĩa là: nhân `value` với $m$, rồi dịch phải $n + \text{value.frac} - \text{frac}$ bit có làm
tròn. Kết quả là $\text{value} \cdot m \cdot 2^{-n}$ ở định dạng Q.`frac`.

**Vì sao chỉ cần dịch theo $n$ mà vẫn đúng, dù scale còn có $m$?** Vì $m$ đã được **nhân thẳng
vào** số nguyên trước khi dịch. Theo dõi đơn vị của số nguyên qua từng bước:

| Bước | Số nguyên | Một đơn vị bằng |
|---|---|---|
| Ban đầu | `value` | $m \cdot 2^{-n}$ (có $m$ lẻ) |
| Nhân với $m$ | `value · m` | $2^{-n}$ (đã sạch $m$) |
| Dịch phải $n - 16$ | kết quả | $2^{-16}$, tức Q.16 |

Phép nhân với $m$ là chính xác (số nguyên nhân số nguyên). Sai số duy nhất là phép dịch, tối
đa ½ LSB.

### Ví dụ

```python
acc = Fixed.const(-26375)
scale = Scale.const(76, 12) * Scale.const(80, 16)   # m = 6080, n = 28
y = scale.rescale(acc, frac=16)
# (-26375 * 6080) >> 12 = -160 360 000 >> 12 = -39150   (Q.16)
y.to_float()                                         # -0.5974
```

Kiểm tra bằng số thực: $-26375 \times 6080 \times 2^{-28} = -0.5974$. Khớp.

## 4.6. `QTensor`

`QTensor(codes, scale)` ghép **mã số nguyên** (thường INT8) với **scale** của nó. Đây là dạng
dữ liệu đi vào mọi phép nhân ma trận. `dequantize()` trả lại số thực, chỉ dùng để debug.

---

← [[03 - Kiểu dữ liệu phần cứng Fixed]] | Tiếp: [[05 - Cách lưu số]] →
