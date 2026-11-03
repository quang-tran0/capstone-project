---
tags: [minilm, fpga, rsqrt, reciprocal]
---

# Chương 13 — rsqrt và nghịch đảo

← [[12 - Mean pooling và cosine]] | Tiếp: [[14 - Độ rộng bit và đưa lên FPGA]] →

Code: `model/numpy/utils/rsqrt.py`, `model/numpy/utils/reciprocal.py`

Mô hình cần hai phép không có sẵn trên FPGA:

| Phép | Dùng ở |
|---|---|
| $1/\sqrt{v}$ (rsqrt) | LayerNorm ([[10 - LayerNorm]]), cosine ([[12 - Mean pooling và cosine]]) |
| $1/v$ (nghịch đảo) | softmax: $1/(127E)$ ([[09 - Self-attention và softmax]]), mean pooling: $1/c$ |

Cả hai dùng chung một cách: **chuẩn hoá → tra bảng → một bước Newton**. Kết quả trả về là một
`Scale` $y \cdot 2^{-s}$ (mantissa $y$ 17 bit), nên dùng thẳng được với `rescale`.

Cả hai chỉ chạy **một lần cho mỗi hàng** (mỗi token hoặc mỗi câu), nên không cần nhanh.

## 13.1. Chuẩn hoá: thu khoảng giá trị về một khoảng nhỏ

$v$ có thể là bất kỳ số nào từ 1 đến khoảng $2^{47}$. Không thể làm bảng cho cả khoảng đó. Cách
làm giống số dấu phẩy động: tách $v$ thành mantissa và số mũ.

**Với rsqrt:** chọn số mũ $E$ **chẵn** để $E/2$ là số nguyên:

$$
v = m \cdot 2^{E}, \quad m \in [1, 4)
\qquad\Rightarrow\qquad
\frac{1}{\sqrt{v}} = \frac{1}{\sqrt{m}} \cdot 2^{-E/2}
$$

**Với nghịch đảo:**

$$
v = m \cdot 2^{p}, \quad m \in [1, 2)
\qquad\Rightarrow\qquad
\frac{1}{v} = \frac{1}{m} \cdot 2^{-p}
$$

Phần $2^{-E/2}$ hay $2^{-p}$ chỉ là dịch bit. Phần khó còn lại là $1/\sqrt{m}$ hoặc $1/m$ với $m$
trong một khoảng nhỏ.

Số mũ tìm bằng **bộ mã hoá ưu tiên** (`leading_one`): vị trí bit 1 cao nhất. Ví dụ $v$ có 40 bit
thì $E = 38$ (số chẵn $\le 39$). $m$ lấy bằng một phép dịch, lưu ở Q.16.

## 13.2. Tra bảng

Các bit đầu của $m$ (sau bit 1 dẫn đầu) là địa chỉ bảng. Mỗi ô lưu giá trị tại **tâm** của khoảng
nó đại diện, để sai số chia đều hai phía.

| | rsqrt | nghịch đảo |
|---|---|---|
| Khoảng của $m$ | $[1, 4)$ | $[1, 2)$ |
| Bước | $1/256$ | $1/256$ |
| Số ô | 768 | 256 |
| Giá trị | $2^{16} / \sqrt{m}$ | $2^{16} / m$ |
| Khoảng giá trị | 32 776 – 65 472 | 32 800 – 65 408 |

## 13.3. Một bước Newton

Bảng cho sai số tương đối khoảng $10^{-3}$. Một bước Newton giảm xuống dưới $10^{-4}$, chỉ bằng
phép nhân và dịch.

**rsqrt** — Newton cho $f(y) = y^{-2} - m$:

$$
y \leftarrow \frac{y \cdot (3 - m y^2)}{2}
$$

Trên số nguyên (Q.16):

$$
u = (y^2) \gg 16, \qquad t = (m \cdot u) \gg 16, \qquad y \leftarrow \big(y \cdot (3 \cdot 2^{16} - t)\big) \gg 17
$$

**Nghịch đảo** — Newton cho $f(y) = 1/y - m$:

$$
y \leftarrow y \cdot (2 - m y)
$$

$$
t = (m \cdot y) \gg 16, \qquad y \leftarrow \big(y \cdot (2 \cdot 2^{16} - t)\big) \gg 16
$$

Newton hội tụ bậc hai: mỗi bước gấp đôi số bit đúng. Từ khoảng 10 bit đúng (bảng) lên khoảng
17 bit, vừa đủ cho mantissa 17 bit.

## 13.4. Kết quả

$$
\frac{1}{\sqrt{v}} \approx y \cdot 2^{-(16 + E/2)}, \qquad \frac{1}{v} \approx y \cdot 2^{-(16 + p)}
$$

Nếu input có phần thập phân ($\text{frac} = f$), hàm tự trừ đi trong số mũ: ví dụ
$1/(\text{raw} \cdot 2^{-f}) = 2^{f} / \text{raw}$.

**Ví dụ rsqrt với $v = 1000$:**

| Bước | Giá trị |
|---|---|
| $E$ | 8 (vì $1000 = 3.906 \cdot 2^8$) |
| $m$ | 3.906 (Q.16: 256 000) |
| Địa chỉ bảng | 744 |
| $y_0$ (bảng) | 33 151 |
| $y_1$ (sau Newton) | 33 159 |
| $y_1 \cdot 2^{-20}$ | 0.0316229 |
| $1/\sqrt{1000}$ chính xác | 0.0316228 |

**Ví dụ trong LayerNorm** ([[10 - LayerNorm#Bước 5 - tính nghịch đảo căn của S]]):
$S = 718\,860\,868\,128$ cho $1/\sqrt{S} \approx 40525 \cdot 2^{-35}$.

## 13.5. Độ chính xác

| Hàm | Sai số tương đối lớn nhất |
|---|---|
| rsqrt | $3.6 \times 10^{-5}$ |
| nghịch đảo | $2.9 \times 10^{-5}$ |

Kiểm thử trên 100 000 giá trị ngẫu nhiên tới $2^{47}$ (rsqrt) và $2^{40}$ (nghịch đảo).

---

← [[12 - Mean pooling và cosine]] | Tiếp: [[14 - Độ rộng bit và đưa lên FPGA]] →
