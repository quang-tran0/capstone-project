---
tags: [minilm, fpga, linear]
---

# Chương 7 — Linear (nhân ma trận)

← [[06 - Lượng tử hoá]] | Tiếp: [[08 - Embedding]] →

Code: `model/numpy/layers/linear.py`

`Linear` chiếm gần như toàn bộ khối lượng tính toán của MiniLM: mỗi encoder layer có 6 phép
(Q, K, V, attention output, FFN dense1, FFN dense2). Trên FPGA, đây là việc của **systolic array**.

## 7.1. Từ công thức gốc đến phép MAC INT8

Công thức gốc, với $K = 384$ hoặc $1536$:

$$
y_{ij} = \sum_{k=1}^{K} x_{ik} W_{kj} + b_j
$$

Thay $x_{ik} \approx x_{q,ik} \cdot S_{x,i}$ (scale theo token) và
$W_{kj} \approx W_{q,kj} \cdot S_{w,j}$ (scale theo cột). Hai scale không phụ thuộc $k$ nên đưa
được ra ngoài tổng:

$$
y_{ij} \approx S_{x,i}\, S_{w,j} \underbrace{\sum_{k} x_{q,ik}\, W_{q,kj}}_{\mathrm{acc}_{ij}} + b_j
$$

Vậy systolic array **chỉ làm phép nhân-cộng INT8**:

$$
\mathrm{acc}_{ij} = \sum_{k=1}^{K} x_{q,ik}\, W_{q,kj}
$$

Phần còn lại (nhân scale, cộng bias) làm ở đầu ra của mảng, một lần cho mỗi phần tử kết quả.

## 7.2. Đổi kết quả về Q.16

Với $S = m \cdot 2^{-n}$:

$$
S_{x,i} S_{w,j} = (m_{x,i}\, m_{w,j}) \cdot 2^{-(n_{x,i} + n_{w,j})}
$$

Muốn kết quả ở Q.16 (nhân thêm $2^{16}$):

$$
[y_{ij}] = \operatorname{round\_shift}\!\Big(\mathrm{acc}_{ij} \cdot (m_{x,i}\, m_{w,j}),\ n_{x,i} + n_{w,j} - 16\Big) + [b_j]
$$

Trong code chỉ là một dòng:

```python
output = (x.scale * self.weight.scale).rescale(accumulator, FRAC_BITS) + self.bias
```

Lý do chỉ cần nhân $m$ rồi dịch theo $n$ đã giải thích ở [[04 - Scale m·2^-n#Rescale - nhân với scale và đổi định dạng]].

## 7.3. Cả quy trình

```python
def __call__(self, x):                                   # x: Q.16 (B, L, K)
    x = quantize(x, axis=-1)                             # mã INT8 + scale theo token
    acc = (x.codes @ self.weight.codes).resize(32)       # systolic array
    y = (x.scale * self.weight.scale).rescale(acc, 16)   # × m, dịch n
    return (y + self.bias).resize(32)                    # Q.16
```

1. **Lượng tử hoá đầu vào** theo token ([[06 - Lượng tử hoá#6.3. Activation (lúc chạy)]]).
2. **MAC INT8** trên systolic array, kết quả vào thanh ghi 32 bit.
3. **Rescale:** nhân $m_x m_w$ (14 bit), dịch phải có làm tròn.
4. **Cộng bias** Q.16, lưu vào thanh ghi 32 bit.

Trong self-attention, $Q$, $K$, $V$ dùng chung **một lần** lượng tử hoá đầu vào.

## 7.4. Ví dụ đầy đủ một phần tử

Phép $X \cdot W_Q$ ở layer 0, câu "Xin chào FPGA" (7 token), phần tử $Y[\text{token } 1, \text{kênh } 0]$:

**Phép này dùng bao nhiêu scale?** $7 + 384 = 391$ cặp $(m, n)$:

| Toán hạng | Shape | Scale | Số scale |
|---|---|---|---|
| $X$ (đầu vào) | $(1, 7, 384)$ | mỗi token một scale | 7 |
| $W_Q$ | $(384, 384)$ | mỗi cột một scale | 384 |
| $Y$ (đầu ra) | $(1, 7, 384)$ | không có, vì là Q.16 | 0 |

**Scale của $X$** (tính lúc chạy):

```
x_m = [119  76  92  86  84  68 116]
x_n = [ 11  12  12  11  12  12  11]
```

**Scale của $W_Q$** (tính offline): kênh 0 có $m = 80$, $n = 16$, tức $S_w = 80 \cdot 2^{-16} = 0.0012207$.

**Tính:**

```
acc   = Σ x_q[1,k] · w_q[k,0]                 = -26375       (cộng 384 tích INT8)
m     = x_m[1] · w_m[0] = 76 · 80             = 6080         (7 bit × 7 bit = 13 bit)
shift = x_n[1] + w_n[0] − 16 = 12 + 16 − 16  = 12
y     = round_shift(-26375 · 6080, 12) + bias
      = -39150 + (-1202)                       = -40352       (Q.16)  →  -0.61572
```

Kiểm tra bằng số thực: $-26375 \times 0.018555 \times 0.0012207 - 0.01834 = -0.6157$. Khớp.

Phần tử $Y[i, j]$ ghép scale của **hàng $i$** với scale của **cột $j$**. Với mỗi phần tử kết quả,
phần cứng cần: 1 phép nhân $m_x m_w$, 1 phép nhân với acc, 1 phép dịch, 1 phép cộng bias.

## 7.5. Độ rộng bit

$$
|\mathrm{acc}| \le K \cdot 127^2 < \begin{cases} 2^{23} & K = 384 \\ 2^{25} & K = 1536 \end{cases}
$$

| Tín hiệu | Độ rộng |
|---|---|
| $x_q$, $W_q$ | 8 bit |
| $\mathrm{acc}$ | thanh ghi 32 bit (dữ liệu $\le 26$ bit) |
| $m_x m_w$ | 14 bit |
| $\mathrm{acc} \cdot m_x m_w$ | $\le 40$ bit (vừa một DSP48E2: 25 × 14 bit, P 48 bit) |
| bias | Q.16, 18 bit là đủ |
| $[y]$ | thanh ghi 32 bit, Q.16 |

> [!warning] Chú ý khi viết RTL
> Tích $\mathrm{acc} \cdot m$ trước khi dịch cần khoảng 40 bit, dù kết quả cuối vừa 32 bit. Bộ
> nhân chỉ ra 32 bit sẽ bị tràn.

---

← [[06 - Lượng tử hoá]] | Tiếp: [[08 - Embedding]] →
