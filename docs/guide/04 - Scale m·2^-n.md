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

| Phép                        | Ý nghĩa                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `S1 * S2`                   | $m = m_1 m_2$, $n = n_1 + n_2$; giữ đủ bit, không làm tròn   |
| `S.rescale(value, frac)`    | Gắn scale $S$ vào số nguyên `value`: trả về $\text{value} \cdot m \cdot 2^{-n}$ ở Q.`frac` (xem bên dưới) |
| `S[idx]`, `S.swapaxes(...)` | Chọn hàng, đổi trục như mảng                                 |
| `S.pack()`                  | Ghép $\{m, n\}$ thành một word (12 bit với mặc định)         |
| `Scale.const(m, n)`         | Hằng số, ví dụ $1/\sqrt{32} \approx$ `Scale.const(5793, 15)` |

### Nhân hai scale

$$
(m_1 2^{-n_1}) \cdot (m_2 2^{-n_2}) = (m_1 m_2) \cdot 2^{-(n_1 + n_2)}
$$

Trên phần cứng là một bộ nhân 7 × 7 bit (ra 14 bit) và một bộ cộng 5 bit.

### Rescale - nhân với scale và đổi định dạng

#### `value` là gì

`rescale` luôn có **hai thứ khác nhau** tham gia:

| Thứ                | Là gì                                                               | Ví dụ                                                         |
| ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------- |
| `value`            | **Dữ liệu**: một số nguyên (hoặc mảng số nguyên) chưa có nghĩa thực | Accumulator sau phép nhân ma trận, mã INT8 của bảng embedding |
| `self` (scale $S$) | **Hệ số** cho biết mỗi đơn vị của `value` đáng giá bao nhiêu        | $S = m \cdot 2^{-n}$                                          |

$m$ và $n$ là của **scale**, không phải của `value`. Giá trị thật mà `value` biểu diễn là:

$$
\text{giá trị thật} = \text{value} \times S = \text{value} \times m \times 2^{-n}
$$

Có thể hình dung như hoá đơn: `value` là **số lượng** (ví dụ 26 375 món), còn $S$ là **đơn giá**
của một món. Muốn biết tổng tiền thì nhân hai thứ với nhau. `rescale` làm đúng phép nhân đó,
và trả kết quả về dạng Q.16 để các bước sau dùng chung một đơn vị.

Ví dụ trong `Linear` ([[07 - Linear]]):

- Systolic array cho ra $\text{acc} = -26375$. Đây là tổng các tích INT8 × INT8, một số nguyên
  không có đơn vị rõ ràng.
- Scale tương ứng là $S = S_x \cdot S_w = 6080 \cdot 2^{-28}$ (tích của scale input và scale weight).
- Giá trị thật là $-26375 \times 6080 \times 2^{-28} = -0.5974$.

Ở đây `value = acc`, còn `m = 6080`, `n = 28` là của scale.

#### Nhiệm vụ của rescale

Đầu vào: số nguyên `value` và scale $(m, n)$.
Đầu ra: **một số nguyên mới ở Q.16** biểu diễn cùng giá trị thật, tức là số nguyên $y$ sao cho

$$
y \cdot 2^{-16} \approx \text{value} \cdot m \cdot 2^{-n}
\qquad\Longleftrightarrow\qquad
y \approx \text{value} \cdot m \cdot 2^{16 - n}
$$

Vế phải chỉ có hai phép:

1. **Nhân `value` với $m$:** nhân số nguyên, chính xác hoàn toàn.
2. **Nhân với $2^{16 - n}$:** tức là **dịch phải $n - 16$ bit** (có làm tròn). Nếu $n < 16$ thì
   là dịch trái.

$$
y = \operatorname{round\_shift}\big(\text{value} \cdot m,\ n - 16\big)
$$

#### Vì sao chỉ cần dịch theo $n$ mà vẫn đúng

Vì $m$ đã được **nhân thẳng vào `value`** ở bước 1, nên sau bước 1 không còn phần $m$ nào phải
xử lý. Theo dõi "một đơn vị của số nguyên đáng giá bao nhiêu" qua từng bước, với ví dụ trên:

| Bước | Số nguyên đang cầm | Một đơn vị của nó đáng giá | Giá trị thật |
|---|---|---|---|
| Ban đầu | `value` = −26 375 | $S = 6080 \cdot 2^{-28}$ | $-26375 \times 6080 \times 2^{-28} = -0.5974$ |
| Sau bước 1: × $m$ | `value · m` = −160 360 000 | $2^{-28}$ | $-160\,360\,000 \times 2^{-28} = -0.5974$ |
| Sau bước 2: dịch phải $28 - 16 = 12$ | $y$ = −39 150 | $2^{-16}$ (Q.16) | $-39150 \times 2^{-16} = -0.5974$ |

Cột cuối không đổi: mỗi bước chỉ đổi cách viết, không đổi giá trị. Bước 1 chuyển $m$ từ "đơn
giá" sang "số lượng", để đơn giá chỉ còn là lũy thừa của 2. Đổi giữa các lũy thừa của 2 thì chỉ
cần dịch bit.

Bước 1 là chính xác (số nguyên nhân số nguyên). Sai số duy nhất nằm ở bước 2, tối đa ½ LSB của
Q.16 nhờ làm tròn.

#### Tham số `frac` và `value.frac`

Code tổng quát hơn công thức trên một chút:

```python
def rescale(self, value, frac):
    return (value * self.m).round_shift(self.n + (value.frac - frac), frac=frac)
```

| Tham số | Ý nghĩa |
|---|---|
| `frac` | Định dạng muốn nhận: Q.`frac`. Hầu hết là 16; cosine dùng 15; LayerNorm dùng 17 |
| `value.frac` | Nếu bản thân `value` đã là số fixed-point (không phải số nguyên thuần) thì phần thập phân này cũng được tính vào lượng dịch |

Lượng dịch tổng quát là $n + \text{value.frac} - \text{frac}$. Với accumulator hay mã INT8 thì
`value.frac = 0` và `frac = 16`, nên lượng dịch là $n - 16$ như ở trên.

#### Ví dụ bằng code

```python
acc = Fixed.const(-26375)                            # value: số nguyên, frac = 0
scale = Scale.const(76, 12) * Scale.const(80, 16)   # S_x * S_w: m = 6080, n = 28
y = scale.rescale(acc, frac=16)
# bước 1: -26375 * 6080      = -160 360 000
# bước 2: round_shift(..., 28 + 0 - 16 = 12) = -39150     (Q.16)
y.to_float()                                         # -0.5974
```

#### Rescale được dùng ở đâu

| Chỗ dùng | `value` | Scale $(m, n)$ |
|---|---|---|
| `Linear` ([[07 - Linear]]) | Accumulator INT8 × INT8 | $S_x \cdot S_w$: scale của token × scale của cột weight |
| Embedding ([[08 - Embedding]]) | Mã INT8 của một hàng bảng | Scale của hàng đó |
| Điểm attention ([[09 - Self-attention và softmax]]) | Accumulator $QK^\top$ | $S_Q \cdot S_K \cdot (1/\sqrt{32})$ |
| Context ([[09 - Self-attention và softmax]]) | Accumulator $PV$ | $S_V \cdot \frac{1}{127E}$ |
| Mean pooling ([[12 - Mean pooling và cosine]]) | Tổng các token (Q.16) | $1/c$ |
| LayerNorm ([[10 - LayerNorm]]) | $e$ đã thu nhỏ | $1/\sqrt{S}$ |

Mọi chỗ đều cùng một mẫu: **có một số nguyên cần "gắn đơn vị", và một scale nói đơn vị đó là
bao nhiêu**.

## 4.6. `QTensor`

`QTensor(codes, scale)` ghép **mã số nguyên** (thường INT8) với **scale** của nó. Đây là dạng
dữ liệu đi vào mọi phép nhân ma trận. `dequantize()` trả lại số thực, chỉ dùng để debug.

---

← [[03 - Kiểu dữ liệu phần cứng Fixed]] | Tiếp: [[05 - Cách lưu số]] →
