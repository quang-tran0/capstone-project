---
tags: [minilm, fpga, attention, softmax]
---

# Chương 9 — Self-attention và softmax

← [[08 - Embedding]] | Tiếp: [[10 - LayerNorm]] →

Code: `model/numpy/layers/self_attention.py`, `model/numpy/utils/softmax.py`

## 9.1. Công thức gốc

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{D}} + M\right) V
$$

$M$ là mask: $-\infty$ ở các token padding để chúng không được chú ý tới. MiniLM có 12 head, mỗi
head làm phép này trên 32 chiều.

Có ba phần khó với số nguyên: nhân $1/\sqrt{32}$, hàm `exp` trong softmax, và phép chia cho tổng
trong softmax. Chương này giải quyết lần lượt.

## 9.2. Q, K, V

Đầu vào $[h]$ (Q.16) được lượng tử hoá **một lần** theo token, rồi dùng chung cho ba phép
`Linear` ([[07 - Linear]]). Kết quả $[Q], [K], [V]$ ở Q.16 được tách thành 12 head, rồi **lượng tử
hoá lại**:

- $Q$, $K$: mỗi token (trong mỗi head) một scale.
- $V$: mỗi kênh (trong mỗi head) một scale, vì bước context cộng dồn theo token
  (xem [[06 - Lượng tử hoá#6.4. Scale theo chiều nào]]).

## 9.3. Điểm attention: gộp $1/\sqrt{D}$ vào scale

$$
\mathrm{sacc}_{ij} = \sum_{d=1}^{32} Q_{q,id}\, K_{q,jd}, \qquad |\mathrm{sacc}| \le 32 \cdot 127^2 < 2^{19}
$$

Thanh ghi 20 bit có dấu là đủ. Điểm thật là

$$
z_{ij} = \mathrm{sacc}_{ij} \cdot S^Q_i \cdot S^K_j \cdot \frac{1}{\sqrt{32}}
$$

Hằng $1/\sqrt{32}$ cũng được viết dạng $m \cdot 2^{-n}$: `Scale.const(5793, 15)`, vì
$5793 / 2^{15} = 0.17679 \approx 1/\sqrt{32}$. Như vậy cả ba hệ số chỉ là tích ba scale:

```python
score_scale = q.scale * k.scale.swapaxes(-2, -1) * INV_SQRT_HEAD_DIM
scores = score_scale.rescale(score_accumulator, 16)        # Q.16
```

$$
[z_{ij}] = \operatorname{round\_shift}\!\Big(\mathrm{sacc}_{ij} \cdot m^Q_i\, m^K_j \cdot 5793,\ \ n^Q_i + n^K_j + 15 - 16\Big)
$$

Tích lớn nhất là $20 + 7 + 7 + 13 = 47$ bit.

## 9.4. Softmax số nguyên

$$
p_{ij} = \frac{e^{z_{ij}}}{\sum_{j'} e^{z_{ij'}}} = \frac{e^{z_{ij} - z_{\max}}}{\sum_{j'} e^{z_{ij'} - z_{\max}}}
$$

### Bước 1 — Trừ max

$$
[t_{ij}] = \max_{j'} [z_{ij'}] - [z_{ij}] \ge 0 \qquad \text{(chỉ xét token có mask = 1)}
$$

Sau bước này mọi số mũ đều $\le 0$, nên $e^{-t} \in (0, 1]$, không bao giờ tràn.

### Bước 2 — Đổi $e$ sang 2

Vì $e^{-t} = 2^{-t \log_2 e}$, nhân với hằng $\mathrm{LOG2E} = \operatorname{round}(2^{15} \log_2 e) = 47\,274$ (Q.15):

$$
[w] = \operatorname{round\_shift}\!\left([t] \cdot \mathrm{LOG2E},\ 15\right) \qquad \text{(Q.16)}
$$

Lý do đổi sang cơ số 2: lũy thừa của 2 với số mũ nguyên chỉ là phép dịch bit.

### Bước 3 — Tách phần nguyên và phần lẻ

$$
w_I = [w] \gg 16, \qquad w_F = [w] \mathbin{\&} (2^{16} - 1), \qquad
2^{-w} = 2^{-w_F / 2^{16}} \cdot 2^{-w_I}
$$

- $2^{-w_I}$: dịch phải $w_I$ bit.
- $2^{-w_F/2^{16}}$ với phần lẻ trong $[0, 1)$: tra bảng.

### Bước 4 — Tra bảng và dịch

Bảng 257 ô ở Q.15, phủ $[0, 1]$ với bước $1/256$:

$$
\mathrm{EXP2}[k] = \operatorname{round}\!\left(2^{15} \cdot 2^{-k/256}\right), \quad k = 0..256
$$

$$
\varepsilon_{ij} = \begin{cases}
\mathrm{EXP2}\big[\operatorname{round\_shift}(w_F, 8)\big] \gg w_I & w_I \le 15 \text{ và mask} = 1 \\
0 & \text{ngược lại}
\end{cases}
$$

Phần tử lớn nhất có $t = 0$, nên $\varepsilon = 2^{15}$ (bằng 1.0). Nếu $w_I > 15$ thì
$e^{-t} < 2^{-15}$, quá nhỏ, coi như 0.

### Bước 5 — Tổng và mã xác suất

$$
E_i = \sum_j \varepsilon_{ij} \ \ (\text{Q.15}), \qquad
P_{q,ij} = \operatorname{round\_shift}\!\left(127 \cdot \varepsilon_{ij},\ 15\right) \in [0, 127]
$$

$P_q$ là mã INT8: phần tử lớn nhất ứng với 127. Khi đó

$$
p_{ij} = \frac{\varepsilon_{ij}}{E_i} \approx \frac{P_{q,ij}}{127 \cdot E_i}
$$

**Phép chia cho $E_i$ không làm ở từng phần tử**, mà gộp vào bước context dưới đây. Như vậy cả
hàng chỉ cần một phép nghịch đảo.

## 9.5. Context

$$
\mathrm{cacc}_{id} = \sum_{j} P_{q,ij}\, V_{q,jd}, \qquad |\mathrm{cacc}| \le 128 \cdot 127^2 < 2^{21}
$$

Thanh ghi 22 bit có dấu. Giá trị thật:

$$
\mathrm{ctx}_{id} = \sum_j p_{ij} v_{jd} \approx \mathrm{cacc}_{id} \cdot S^V_d \cdot \frac{1}{127 \cdot E_i}
$$

$1/(127 E_i)$ được tính **một lần cho mỗi hàng** bằng `reciprocal_fixed`
([[13 - rsqrt và nghịch đảo]]), trả về một `Scale`. Nhân với scale của $V$ rồi rescale:

```python
inverse_total = reciprocal_fixed(127 * total, self.reciprocal_lut)
context = (v.scale * inverse_total).rescale(context_accumulator, 16)
```

Tích lớn nhất là $22 + 7 + 17 = 46$ bit. 12 head được ghép lại thành vector 384 chiều, lưu vào
thanh ghi 32 bit Q.16.

## 9.6. Tóm tắt các thanh ghi

| Tín hiệu | Độ rộng |
|---|---|
| $Q_q$, $K_q$, $V_q$ | 8 bit |
| $\mathrm{sacc}$ | 20 bit |
| $[z]$ (điểm) | Q.16 |
| $\varepsilon$ | Q.15, 16 bit |
| $E$ (tổng) | $\le 2^{22}$ với 128 token |
| $P_q$ | 8 bit, $[0, 127]$ |
| $\mathrm{cacc}$ | 22 bit |
| context | 32 bit, Q.16 |

---

← [[08 - Embedding]] | Tiếp: [[10 - LayerNorm]] →
