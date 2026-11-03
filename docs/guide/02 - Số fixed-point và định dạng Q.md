---
tags: [minilm, fpga, fixed-point]
---

# Chương 2 — Số fixed-point và định dạng Q

← [[01 - Tổng quan]] | Tiếp: [[03 - Kiểu dữ liệu phần cứng Fixed]] →

Chương này giải thích cách biểu diễn số thực bằng số nguyên, nền tảng của mọi chương sau.

## 2.1. Định dạng Q

Một số nguyên $X$ ở **định dạng Q.$f$** biểu diễn giá trị thực

$$
x = X \cdot 2^{-f}
$$

$f$ gọi là **số bit thập phân** (fraction bits). Có thể hình dung là dấu phẩy được đặt cố định
sau $f$ bit cuối, nên gọi là **fixed-point** (dấu phẩy tĩnh).

| Giá trị thực | Q.16 (số nguyên lưu) | Cách tính |
|---|---|---|
| 1.0 | 65 536 | $1 \cdot 2^{16}$ |
| 1.5 | 98 304 | $1.5 \cdot 2^{16}$ |
| −0.25 | −16 384 | $-0.25 \cdot 2^{16}$ |
| 0.0000153 | 1 | bước nhỏ nhất $2^{-16}$ |

Đồ án dùng **Q.16 trong thanh ghi 32 bit có dấu** cho hầu hết tensor. Khoảng biểu diễn được là
$[-32768, 32768)$ với bước $2^{-16} \approx 1.5 \times 10^{-5}$.

## 2.2. Cộng và nhân trong định dạng Q

**Cộng:** hai số cùng Q.$f$ thì cộng thẳng số nguyên, kết quả vẫn Q.$f$:

$$
X_1 2^{-f} + X_2 2^{-f} = (X_1 + X_2) 2^{-f}
$$

Nếu khác $f$ thì phải dịch cho cùng $f$ trước, giống như phải đặt thẳng dấu phẩy khi cộng tay.

**Nhân:** phần thập phân cộng lại:

$$
(X_1 2^{-f_1}) \cdot (X_2 2^{-f_2}) = (X_1 X_2) \cdot 2^{-(f_1 + f_2)}
$$

Ví dụ: Q.16 × Q.15 cho kết quả Q.31. Muốn quay về Q.16 thì dịch phải 15 bit.

## 2.3. Dịch bit

Dịch phải $k$ bit = chia cho $2^k$. Dịch trái $k$ bit = nhân với $2^k$. Trên FPGA, dịch bit gần
như miễn phí (chỉ là nối dây hoặc một bộ chọn).

Có hai cách hiểu một phép dịch:

| Cách hiểu | Ý nghĩa | Ví dụ |
|---|---|---|
| **Đổi định dạng** | Giá trị thực giữ nguyên, chỉ bớt độ chính xác | Q.31 dịch phải 15 → Q.16 |
| **Chia cho $2^k$** | Định dạng giữ nguyên, giá trị nhỏ đi | Thu nhỏ một vector để vừa 18 bit |

## 2.4. Làm tròn khi dịch

Dịch phải "thô" (`>>`) là **làm tròn xuống** (về $-\infty$): $7 \gg 1 = 3$, $-7 \gg 1 = -4$.
Sai số có thể tới gần 1 LSB, và luôn lệch về một phía.

Đồ án dùng **làm tròn nửa lên** (round half up): cộng thêm một nửa bước trước khi dịch:

$$
\operatorname{round\_shift}(x, k) =
\begin{cases}
\left(x + 2^{k-1}\right) \gg k & k > 0 \\
x & k = 0 \\
x \ll (-k) & k < 0
\end{cases}
$$

Sai số tối đa chỉ còn ½ LSB. Trên phần cứng, việc này chỉ thêm một bộ cộng hằng số.

| $x$ | $x \gg 1$ (làm tròn xuống) | $\operatorname{round\_shift}(x, 1)$ |
|---|---|---|
| 3 | 1 | 2 |
| −3 | −2 | −1 |
| 5 | 2 | 3 |

Ngoài ra có **làm tròn lên** (ceiling): $\lceil x / 2^k \rceil = (x + 2^k - 1) \gg k$, dùng khi
chọn scale (xem [[06 - Lượng tử hoá]]).

## 2.5. Độ rộng bit

Mỗi phép tính làm số bit tăng lên:

| Phép | Cần thêm |
|---|---|
| Cộng hai số $b$ bit | 1 bit |
| Nhân số $a$ bit với số $b$ bit | ra $a + b$ bit |
| Cộng dồn $K$ số | $\lceil \log_2 K \rceil$ bit |

Ví dụ: tích INT8 × INT8 cần 16 bit; cộng dồn 384 tích như vậy cần $16 + 9 = 25$ bit.

Trên FPGA, mỗi thanh ghi có độ rộng cố định. Nếu giá trị vượt quá thì bị **tràn** (overflow):
bit cao bị mất và kết quả sai hoàn toàn. Vì vậy phải biết trước mỗi tín hiệu cần bao nhiêu bit.
Đây là lý do có class `Fixed` ở chương sau.

## 2.6. Số có dấu và không dấu

- **Có dấu** (bù 2), $b$ bit: khoảng $[-2^{b-1}, 2^{b-1} - 1]$. Ví dụ INT8: $[-128, 127]$.
- **Không dấu**, $b$ bit: khoảng $[0, 2^b - 1]$. Ví dụ UINT5: $[0, 31]$.

Các mã INT8 trong đồ án dùng khoảng **đối xứng** $[-127, 127]$, bỏ $-128$, để giá trị tuyệt đối
luôn vừa 7 bit.

---

← [[01 - Tổng quan]] | Tiếp: [[03 - Kiểu dữ liệu phần cứng Fixed]] →
