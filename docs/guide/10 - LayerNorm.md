---
tags: [minilm, fpga, layernorm]
---

# Chương 10 — LayerNorm

← [[09 - Self-attention và softmax]] | Tiếp: [[11 - GELU và residual]] →

Code: `model/numpy/layers/int_layer_norm.py`

LayerNorm xuất hiện 25 lần trong MiniLM (1 sau embedding, 2 trong mỗi encoder layer). Công thức
gốc cần phép chia và căn bậc hai, cả hai đều không có trên đường tính số nguyên. Chương này cho
thấy cách bỏ cả hai.

## 10.1. Công thức gốc

Với một hàng $N = 384$ phần tử:

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i,
\qquad
\mu = \frac{1}{N}\sum_{j} x_j,
\qquad
\sigma^2 = \frac{1}{N}\sum_{j} (x_j - \mu)^2
$$

## 10.2. Ý tưởng chính: bỏ phép chia cho $N$

Đặt

$$
d_i = N x_i - \sum_j x_j = N (x_i - \mu)
$$

$d$ tính được chính xác bằng số nguyên (không chia). Khi đó:

- $\lVert d \rVert = N \sqrt{\sum (x - \mu)^2}$
- $\sigma = \sqrt{\sum (x - \mu)^2 / N}$

Thay vào, các thừa số $N$ rút gọn, chỉ còn $\sqrt{N}$:

$$
\frac{x_i - \mu}{\sigma} = \sqrt{N} \cdot \frac{d_i}{\lVert d \rVert}
\qquad\Rightarrow\qquad
y_i = \underbrace{\left(\gamma_i \sqrt{N}\right)}_{G_i} \cdot \frac{d_i}{\lVert d \rVert} + \beta_i
$$

Hai điều làm phần cứng gọn:

1. **$\sqrt{N}$ là hằng số**, được gộp sẵn vào $\gamma$ ở bước offline: $G = \gamma \cdot \sqrt{384}$,
   lưu ở Q.10, 18 bit.
2. **$d / \lVert d \rVert$ không phụ thuộc scale của input**: nhân cả hàng với một hằng số thì tỉ
   số không đổi. Vì vậy có thể dịch cả hàng đi bao nhiêu bit tuỳ ý mà không sai.

$\varepsilon = 10^{-12}$ được bỏ qua vì rất nhỏ so với phương sai thực tế.

Phần còn lại (bước 3–7 dưới đây) chỉ làm một việc: **tính $d / \lVert d \rVert$ bằng số nguyên**,
rồi nhân $G$ và cộng $\beta$.

## 10.3. Các bước, với số liệu thật

Ví dụ: LayerNorm sau embedding, token thứ 2 của câu "Xin chào". Chỉ ghi 4 phần tử đầu trong 384.

### Bước 1–2: tính $d$

```
x[:4] = 796, -2268, -66, -814          (Q.16 → 0.01215, -0.03461, -0.00101, -0.01242)
d = 384·x − Σx
d[:4] = 327657, -848919, -3351, -290583
```

$384 \cdot x$ có thể làm bằng dịch và cộng: $384 = 256 + 128$, nên $384x = (x \ll 8) + (x \ll 7)$.

### Bước 3: thu nhỏ $d$ để vừa 18 bit

**Vấn đề:** $d$ cần tới 44 bit (giá trị cỡ 1.9 triệu). Bước sau phải bình phương $d$. Nhân thẳng
44 × 44 bit ra 88 bit, không DSP nào chứa được. DSP48E2 chỉ nhân $27 \times 18$ bit.

**Cách làm:** dịch **cả hàng** sang phải cùng $s$ bit, sao cho phần tử lớn nhất vừa đúng 18 bit
có dấu:

```
max|d|            = 1 936 023
leading_one(...)  = 21            (số này cần 21 bit)
s = 21 + 1 − 18   = 4             (+1 cho bit dấu; muốn còn 18 bit → bỏ 4 bit thấp)
e = d >> 4
e[:4] = 20478, -53058, -210, -18162        max|e| = 121002 < 2¹⁷ ✓
```

**Vì sao không làm sai kết quả:** dịch phải 4 bit nghĩa là chia mọi phần tử cho 16, nên
$\lVert d \rVert$ cũng chia cho 16:

$$
\frac{e}{\lVert e \rVert} = \frac{d/16}{\lVert d \rVert / 16} = \frac{d}{\lVert d \rVert}
$$

Chỉ có 4 bit thấp của mỗi phần tử bị mất. Phần tử lớn nhất vẫn giữ đủ 18 bit nên phần mất rất
nhỏ (độ chính xác tương đối khoảng $2^{-17}$). Mỗi hàng có $s$ riêng.

### Bước 4: tổng bình phương

```
S = Σ e²  (384 số)  = 718 860 868 128          (cần 41 bit)
```

Mỗi $e^2$ là nhân $18 \times 18$ bit (tối đa 36 bit). Cộng 384 số thêm khoảng 9 bit, tối đa
45 bit, vừa **accumulator 48 bit** của DSP48E2. Ta có $\lVert e \rVert = \sqrt{S}$.

### Bước 5 - tính nghịch đảo căn của S

Không dùng căn và chia.

Cần $1/\sqrt{718\,860\,868\,128} \approx 1.17944 \times 10^{-6}$. Dùng ba kỹ thuật (chi tiết ở
[[13 - rsqrt và nghịch đảo]]):

**(a) Tách $S$ thành mantissa × lũy thừa của 2**, giống số dấu phẩy động:

```
S có 40 bit  →  số mũ E = 38   (chọn số chẵn)
S = m · 2³⁸    với m = 2.615   (luôn nằm trong [1, 4))
1/√S = 1/√m · 2⁻¹⁹
```

Phải chọn số mũ chẵn để $38/2 = 19$ là số nguyên, khi đó phần lũy thừa của 2 chỉ là dịch bit.
Phần khó thu lại còn $1/\sqrt{m}$ với $m$ trong khoảng nhỏ $[1, 4)$.

**(b) Tra bảng** $1/\sqrt{m}$: 768 ô, chia $[1, 4)$ thành các bước $1/256$. Được $y \approx 0.6184$.

**(c) Một bước Newton** để làm mịn: $y \leftarrow y \cdot (3 - m y^2) / 2$. Sai số tương đối giảm
từ khoảng $10^{-3}$ xuống dưới $10^{-4}$.

**Kết quả** là một `Scale(m, n)`:

```
1/√S ≈ 40525 · 2⁻³⁵ = 1.1794327e-6        (đúng: 1.1794447e-6)
```

($0.6184 \approx 40525 \cdot 2^{-16}$, cộng phần số mũ $2^{-19}$, tổng là $2^{-35}$.)

Bước này chỉ chạy **một lần cho mỗi hàng**, nên trên phần cứng có thể làm tuần tự mà không ảnh
hưởng tốc độ.

### Bước 6: tính $d / \lVert d \rVert = e \cdot (1/\sqrt{S})$

Mỗi phần tử nhân với $m = 40525$ rồi dịch để kết quả ở **Q.17**:

```
norm = (e · 40525) >> (35 − 17) = (e · 40525) >> 18

e[0] = 20478  →  20478·40525 >> 18 = 3166   →  3166 / 2¹⁷ = 0.024155

norm[:4]           = 3166, -8202, -32, -2808  →  0.024155, -0.062576, -0.000244, -0.021423
d/‖d‖ đúng [:4]                                →  0.024153, -0.062578, -0.000247, -0.021420
```

Dịch $35 - 17 = 18$ bit vì giá trị thật là $e \cdot 40525 \cdot 2^{-35}$, và muốn lưu ở Q.17
(nhân thêm $2^{17}$) thì chỉ dịch phải 18 bit.

**Vì sao Q.17:** $d / \lVert d \rVert$ luôn nằm trong $[-1, 1]$ (không phần tử nào lớn hơn độ dài
cả vector). Q.17 cộng bit dấu vừa đúng 18 bit, khớp cổng 18 bit của DSP ở bước 7.

**Bão hoà $\pm(2^{17} - 1)$:** giá trị bằng đúng 1.0 chỉ xảy ra khi cả hàng có duy nhất một phần
tử khác 0. Lúc đó $2^{17}$ không vừa 18 bit có dấu, nên kẹp xuống $2^{17} - 1$.

### Bước 7: nhân $G$, cộng $\beta$

$G = \gamma \sqrt{384}$ được tính sẵn ở Q.10:

```
γ[:4] = 0.8518, 0.8578, 0.7717, 0.7678
G = round(γ · √384 · 2¹⁰) = 17092, 17213, 15485, 15406
```

$$
y = (\text{norm} \cdot G) \gg 11 + \beta
$$

Dịch 11 bit vì: norm có 17 bit thập phân, $G$ có 10, tích có 27; muốn Q.16 thì bỏ $27 - 16 = 11$
bit. $\beta$ đã ở Q.16 nên cộng thẳng.

```
y[0] = (3166 · 17092) >> 11 + (−2012) = 26422 − 2012 = 24410  →  0.372467

y (NumPy) [:4] = 0.372467, -1.158478, 0.026840, -0.394821
LayerNorm float = 0.372452, -1.158512, 0.026797, -0.394778
```

Sai lệch khoảng $10^{-5}$ đến $4 \times 10^{-5}$, tức vài LSB của Q.16.

### Trường hợp đặc biệt

Hàng hằng số ($d = 0$, $S = 0$): bỏ qua rsqrt, kết quả là $\beta$.

## 10.4. Tóm tắt

| Bước | Làm gì | Độ rộng | Vì sao |
|---|---|---|---|
| 1–2 | $d = 384x - \sum x$ | ~44 bit | Căn giữa chính xác, không chia |
| 3 | $e = d \gg s$ | 18 bit | Vừa cổng nhân 18 bit; tỉ số $d/\lVert d\rVert$ không đổi |
| 4 | $S = \sum e^2$ | 48 bit | Có $\lVert e \rVert^2$ trong accumulator DSP |
| 5 | $1/\sqrt{S}$ = bảng + Newton + dịch số mũ | mantissa 17 bit | Không căn, không chia; mỗi hàng một lần |
| 6 | $\text{norm} = e \cdot m \gg (n - 17)$ | 18 bit, Q.17 | Được $d/\lVert d \rVert \in [-1, 1]$ |
| 7 | $y = \text{norm} \cdot G \gg 11 + \beta$ | 32 bit, Q.16 | $\sqrt{N}$ đã gộp vào $G$ |

## 10.5. Trên FPGA

- **Mỗi phần tử chỉ cần 3 phép nhân**: $e^2$ ($18 \times 18$), $e \cdot m$ ($18 \times 17$),
  $\text{norm} \cdot G$ ($18 \times 18$). Mỗi phép vừa một DSP48E2.
- **Cần đệm cả hàng** ($384 \times 33$ bit $\approx$ 1.6 KB) trong BRAM, vì phải quét hàng vài
  lượt: tính $\sum x$; tính $d$ và $\max |d|$; tính $\sum e^2$; cuối cùng chuẩn hoá từng phần tử.
- rsqrt chỉ chạy một lần mỗi hàng nên có thể làm tuần tự, ít tài nguyên.

## 10.6. Độ chính xác

- Test với $\gamma$, $\beta$ ngẫu nhiên và input có kênh outlier lớn gấp 30 lần: sai số lớn nhất
  so với LayerNorm float dưới $2 \times 10^{-3}$, trung bình dưới $2 \times 10^{-4}$.
- Trong báo cáo so sánh, các dòng LayerNorm có sai số trung bình khoảng 0.03–0.06%, thấp hơn hẳn
  các bước xung quanh. LayerNorm còn **kéo sai số tích luỹ xuống lại** sau mỗi khối.

---

← [[09 - Self-attention và softmax]] | Tiếp: [[11 - GELU và residual]] →
