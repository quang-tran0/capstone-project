# Tính toán số nguyên trên FPGA cho MiniLM

Tài liệu này mô tả chi tiết cách mô hình NumPy (`model/numpy`) tính sentence embedding
và score RAG của `paraphrase-multilingual-MiniLM-L12-v2` **hoàn toàn bằng số nguyên**,
theo cách phần cứng trên KV260 sẽ làm.

**Lúc chạy chỉ dùng:** cộng, trừ, nhân số nguyên, dịch bit, so sánh, tra LUT.
**Không dùng:** chia, căn, `exp`, `erf`, số thực.

Mọi phép khó được tính sẵn trên máy tính (offline) thành hằng số hoặc bảng LUT. Các hàm
`prepare_*` và phần `__init__` của mỗi layer là bước offline đó. Hàm `to_float` chỉ dùng để
in trace, không nằm trong đường tính.

## Mục lục

1. [Ký hiệu và phép toán cơ bản](#1-ký-hiệu-và-phép-toán-cơ-bản)
2. [Biểu diễn dữ liệu](#2-biểu-diễn-dữ-liệu)
3. [Lượng tử hoá](#3-lượng-tử-hoá)
4. [Embedding](#4-embedding)
5. [Nhân ma trận `Linear`](#5-nhân-ma-trận-linear)
6. [Self-attention và softmax](#6-self-attention-và-softmax)
7. [Cộng residual](#7-cộng-residual)
8. [LayerNorm](#8-layernorm)
9. [GELU](#9-gelu)
10. [Mean pooling](#10-mean-pooling)
11. [Score RAG: cosine](#11-score-rag-cosine)
12. [Căn bậc 2 nghịch đảo và nghịch đảo](#12-căn-bậc-2-nghịch-đảo-và-nghịch-đảo)
13. [Bảng hằng số, LUT và độ rộng bit](#13-bảng-hằng-số-lut-và-độ-rộng-bit)
14. [Độ chính xác](#14-độ-chính-xác)

---

## 1. Ký hiệu và phép toán cơ bản

| Ký hiệu | Ý nghĩa |
|---|---|
| $N = 384$ | Hidden size |
| $H = 12,\ D = 32$ | Số head, số chiều mỗi head |
| $F = 1536$ | Kích thước tầng trung gian của FFN |
| $L \le 128$ | Số token của câu |
| $\mathrm{Q}.f$ | Số nguyên $X$ biểu diễn giá trị thực $X / 2^{f}$ |
| $[x]$ | Giá trị Q.16 của $x$, tức $\operatorname{round}(x \cdot 2^{16})$ |

**Dịch phải số học** (bỏ bit thấp):

$$
x \gg k = \left\lfloor x / 2^{k} \right\rfloor
$$

**Dịch phải có làm tròn** (`round_shift`); khi $k < 0$ thì đây là dịch trái:

$$
\operatorname{rshift}(x, k) =
\begin{cases}
\left(x + 2^{k-1}\right) \gg k & k > 0 \\
x & k = 0 \\
x \ll (-k) & k < 0
\end{cases}
$$

**Dịch phải làm tròn lên:**

$$
\operatorname{cshift}(x, k) = \left(x + 2^{k} - 1\right) \gg k = \left\lceil x / 2^{k} \right\rceil
$$

**Bộ mã hoá ưu tiên** (`bit_length`), tức mạch tìm bit 1 cao nhất:

$$
\operatorname{bitlen}(v) = \begin{cases} \lfloor \log_2 v \rfloor + 1 & v > 0 \\ 0 & v = 0 \end{cases}
$$

**Bão hoà** về $B$ bit có dấu:

$$
\operatorname{sat}_B(x) = \min\!\left(\max\!\left(x,\ -(2^{B-1}-1)\right),\ 2^{B-1}-1\right)
$$

**Kiểm tra tràn** (`check_bits`): mô phỏng báo `OverflowError` nếu
$x \notin [-2^{B-1}, 2^{B-1})$. Tức là mỗi thanh ghi trên FPGA đều được kiểm tra độ rộng.

---

## 2. Biểu diễn dữ liệu

### 2.1. Hai kiểu tensor

| Kiểu | Định dạng | Dùng cho |
|---|---|---|
| **Rộng** | Q.16 trong INT32 | Residual stream, đầu ra `Linear`, LayerNorm, GELU, pooling |
| **Hẹp** | INT8 kèm scale $S = m \cdot 2^{-n}$ | Đầu vào của mọi phép nhân ma trận |

Tensor rộng dùng chung một đơn vị $2^{-16}$, nên cộng residual chỉ là phép cộng số nguyên.
Tensor hẹp chỉ xuất hiện ngay trước systolic array.

### 2.2. Scale

$$
S = m \cdot 2^{-n}, \qquad m \in [1, 127]\ (\text{INT8}), \quad n \in [0, 31]\ (\text{UINT5})
$$

Đây là định dạng `ScaleInt12` với $n$ mở rộng từ 4 lên **5 bit**. Lý do: scale per-channel
của weight thường cỡ $10^{-3}$–$10^{-4}$, nhỏ hơn $2^{-15}$. Nếu giới hạn $n \le 15$ thì $m$
chỉ còn vài đơn vị và mất độ chính xác.

### 2.3. Vì sao dùng scale per-token và per-channel

Mình thử chèn lượng tử hoá INT8 giả lập (fake-quant) vào model PyTorch float32 và đo cosine
của sentence embedding so với bản gốc, trên 6 câu:

| Phương án | Cosine nhỏ nhất | Độ lệch score lớn nhất |
|---|---|---|
| INT8 per-tensor ở mọi chỗ (thiết kế cũ) | 0.849 | 0.119 |
| Giữ đầu ra `Linear` ở dạng rộng | 0.920 | 0.109 |
| + weight per-channel | 0.958 | 0.052 |
| **+ activation per-token** | **0.998** | **0.010** |

Activation của transformer có vài kênh outlier rất lớn. Với một scale cho cả tensor, các
giá trị bình thường chỉ dùng được vài mức INT8. Khi mỗi token có scale riêng và mỗi cột
weight có scale riêng, toàn bộ dải INT8 được dùng hết.

---

## 3. Lượng tử hoá

### 3.1. Weight (offline, `quantize_weight`)

Mỗi lát cắt (một cột của weight `Linear`, hoặc một hàng của bảng embedding) có scale riêng.
Chọn $n$ lớn nhất để $m$ vừa INT8, và **làm tròn $m$ lên**:

$$
s = \frac{\max |W_{:,j}|}{127}, \qquad
n_j = \max\left\{ n \le 31 \;\middle|\; \lceil s \cdot 2^{n} \rceil \le 127 \right\}, \qquad
m_j = \left\lceil s \cdot 2^{n_j} \right\rceil
$$

$$
W_{q,kj} = \operatorname{clip}\!\left(\operatorname{rint}\!\left(\frac{W_{kj}}{m_j 2^{-n_j}}\right), -127, 127\right)
$$

Làm tròn lên bảo đảm $m_j 2^{-n_j} \ge s$, nên phần tử lớn nhất cho mã $\le 127$: không
phần tử nào bị cắt, và sai số luôn $\le \tfrac{1}{2}$ bước.

### 3.2. Activation (lúc chạy, `quantize_fixed`)

Đầu vào là một hàng $x$ ở Q.16. Tất cả đều là phép số nguyên.

**Bước 1: tìm đỉnh.**

$$
P = \max_k |x_k| \qquad \text{(Q.16)}
$$

**Bước 2: chia cho 127 bằng phép nhân.** Dùng hằng số $R_{127} = \lceil 2^{24} / 127 \rceil = 132\,105$:

$$
P \cdot R_{127} \approx \frac{P}{127} \cdot 2^{24} \qquad \text{(Q.40)}
$$

**Bước 3: chọn $n$** bằng bộ mã hoá ưu tiên trên 32 ứng viên:

$$
m(n) = \operatorname{cshift}\!\left(P \cdot R_{127},\ 40 - n\right), \qquad
n^{\star} = \max\{\, n \mid m(n) \le 127 \,\}, \qquad m = m(n^{\star})
$$

Nếu $P = 0$ thì đặt $m = 1$, $n = 31$.

**Bước 4: tính mã.** $x / S = x \cdot 2^{-16} \cdot 2^{n} / m$. Phép chia cho $m$ được thay
bằng phép nhân với LUT nghịch đảo 127 ô:

$$
\mathrm{RECIP}[m] = \operatorname{round}\!\left(\frac{2^{22}}{m}\right), \qquad
x_{q,k} = \operatorname{sat}_8\!\Big(\operatorname{rshift}\!\left(x_k \cdot \mathrm{RECIP}[m],\ 22 + 16 - n\right)\Big)
$$

Vì $m \in [64, 127]$ (khi $n$ chưa chạm giới hạn), $\mathrm{RECIP}[m] \in [33\,026,\ 65\,536]$.

### 3.3. Chiều của scale

| Tensor | Scale theo | Hình dạng của $(m, n)$ |
|---|---|---|
| Weight `Linear` | cột (kênh ra) | $(1, \text{out})$ |
| Bảng embedding | hàng (token ID) | $(V, 1)$ |
| Đầu vào `Linear` | token | $(B, L, 1)$ |
| $Q$, $K$ | token × head | $(B, H, L, 1)$ |
| $V$ | kênh × head | $(B, H, 1, D)$ |

$V$ dùng scale theo kênh vì context cộng dồn qua các token. Nếu scale theo token thì không
đưa được scale ra ngoài tổng.

---

## 4. Embedding

**Offline:** ba bảng (word $250\,037 \times 384$, position $512 \times 384$, token type
$2 \times 384$) được lượng tử hoá INT8 với scale theo hàng (mục 3.1).

**Lúc chạy:** tra một hàng và đổi về Q.16:

$$
[e_k] = \operatorname{rshift}\!\left(q_k \cdot m,\ n - 16\right)
$$

với $q_k$ là mã INT8, còn $m$, $n$ là scale của hàng đó. Phép nhân là $8 \times 7$ bit.
Sau đó:

$$
[x] = [e_{\text{word}}] + [e_{\text{pos}}] + [e_{\text{type}}]
$$

rồi đưa qua LayerNorm (mục 8). Position ID là bộ đếm $0, 1, \dots, L-1$.

---

## 5. Nhân ma trận `Linear`

### Công thức gốc

$$
y_{ij} = \sum_{k=1}^{K} x_{ik} W_{kj} + b_j
$$

với $K = 384$ hoặc $1536$.

### Biến đổi

Thay $x_{ik} \approx x_{q,ik}\, m_i 2^{-n_i}$ (theo token $i$) và
$W_{kj} \approx W_{q,kj}\, m_j 2^{-n_j}$ (theo cột $j$):

$$
y_{ij} \approx (m_i m_j)\, 2^{-(n_i + n_j)} \underbrace{\sum_{k} x_{q,ik} W_{q,kj}}_{\mathrm{acc}_{ij}} + b_j
$$

Scale của token $i$ và của cột $j$ đều nằm ngoài tổng, nên **systolic array chỉ làm MAC INT8**:

$$
\boxed{\ \mathrm{acc}_{ij} = \sum_{k=1}^{K} x_{q,ik}\, W_{q,kj}\ }
$$

**Đổi về Q.16** ở đầu ra của mảng:

$$
\boxed{\ [y_{ij}] = \operatorname{rshift}\!\Big(\mathrm{acc}_{ij} \cdot (m_i\, m_j),\ n_i + n_j - 16\Big) + [b_j]\ }
$$

với $[b_j] = \operatorname{round}(b_j \cdot 2^{16})$ tính sẵn offline. Bias không còn phụ
thuộc scale lúc chạy, nên không cần phép chia.

### Độ rộng bit

$$
|\mathrm{acc}| \le K \cdot 127^2 < \begin{cases} 2^{23} & K = 384 \\ 2^{25} & K = 1536 \end{cases}
$$

| Tín hiệu | Độ rộng |
|---|---|
| $x_q \cdot W_q$ | $8 \times 8$ bit |
| $\mathrm{acc}$ | $\le 26$ bit (kiểm tra ở 32 bit) |
| $m_i m_j$ | 14 bit |
| $\mathrm{acc} \cdot m_i m_j$ | $\le 40$ bit |
| $[y]$ | 32 bit |

---

## 6. Self-attention và softmax

### Công thức gốc

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{D}} + M\right) V
$$

### 6.1. Q, K, V

Đầu vào $[h]$ (Q.16) được lượng tử hoá **một lần** theo token, rồi dùng chung cho ba phép
`Linear` ở mục 5. Kết quả $[Q]$, $[K]$, $[V]$ ở Q.16 được tách thành 12 head, sau đó lượng
tử hoá lại theo bảng ở mục 3.3:

$$
Q_q,\ (m^Q_{i}, n^Q_{i}) \quad\text{theo token } i, \qquad
K_q,\ (m^K_{j}, n^K_{j}) \quad\text{theo token } j, \qquad
V_q,\ (m^V_{d}, n^V_{d}) \quad\text{theo kênh } d
$$

### 6.2. Điểm attention, gộp $1/\sqrt{D}$ vào scale

$$
\mathrm{sacc}_{ij} = \sum_{d=1}^{32} Q_{q,id}\, K_{q,jd}, \qquad |\mathrm{sacc}| \le 32 \cdot 127^2 < 2^{19}
$$

$\mathrm{sacc}$ vừa **20 bit có dấu**. Điểm thực là
$z_{ij} = \mathrm{sacc}_{ij}\, m^Q_i m^K_j\, 2^{-(n^Q_i + n^K_j)} / \sqrt{32}$.
Hằng số $1/\sqrt{32}$ được gộp thành $C = \operatorname{round}(2^{15}/\sqrt{32}) = 5793$:

$$
\boxed{\ [z_{ij}] = \operatorname{rshift}\!\Big(\mathrm{sacc}_{ij} \cdot m^Q_i\, m^K_j \cdot C,\ \ n^Q_i + n^K_j + 15 - 16\Big)\ }
$$

Tích lớn nhất là $20 + 7 + 7 + 13 = 47$ bit.

### 6.3. Softmax số nguyên

$$
p_{ij} = \frac{e^{z_{ij} - z_{\max}}}{\sum_{j'} e^{z_{ij'} - z_{\max}}}
$$

**Bước 1: trừ max.** Với mask $M_j \in \{0, 1\}$:

$$
[z_{\max}] = \max_{j:\,M_j = 1} [z_{ij}], \qquad [t_{ij}] = [z_{\max}] - [z_{ij}] \ge 0
$$

**Bước 2: đổi $e^{-t}$ sang $2^{-w}$.** Vì $e^{-t} = 2^{-t \log_2 e}$, với
$\mathrm{LOG2E} = \operatorname{round}(\log_2 e \cdot 2^{15}) = 47\,274$:

$$
[w] = \operatorname{rshift}\!\left([t] \cdot \mathrm{LOG2E},\ 15\right) \qquad \text{(Q.16)}
$$

**Bước 3: tách phần nguyên và phần lẻ.**

$$
w_I = [w] \gg 16, \qquad w_F = [w] \mathbin{\&} (2^{16} - 1), \qquad
2^{-w} = 2^{-w_F / 2^{16}} \cdot 2^{-w_I}
$$

**Bước 4: tra LUT phần lẻ, dịch theo phần nguyên.** LUT có 257 ô ở Q.15:

$$
\mathrm{EXP2}[k] = \operatorname{round}\!\left(2^{15} \cdot 2^{-k/256}\right), \quad k = 0..256
$$

$$
\boxed{\ \varepsilon_{ij} = \begin{cases}
\mathrm{EXP2}\big[\operatorname{rshift}(w_F, 8)\big] \gg w_I & w_I \le 15 \text{ và } M_j = 1 \\
0 & \text{ngược lại}
\end{cases}\ }
$$

Phần tử lớn nhất có $t = 0$, nên $\varepsilon = 2^{15}$ (bằng 1.0).

**Bước 5: tổng và mã xác suất.**

$$
E_i = \sum_j \varepsilon_{ij} \quad (\le 128 \cdot 2^{15} = 2^{22}), \qquad
P_{q,ij} = \operatorname{rshift}\!\left(127 \cdot \varepsilon_{ij},\ 15\right) \in [0, 127]
$$

Khi đó

$$
p_{ij} = \frac{\varepsilon_{ij}}{E_i} \approx \frac{P_{q,ij} \cdot 2^{15}}{127 \cdot E_i}
$$

Phép chia cho $E_i$ **không** làm ở từng phần tử, mà gộp vào bước context.

### 6.4. Context

$$
\mathrm{cacc}_{id} = \sum_{j=1}^{L} P_{q,ij}\, V_{q,jd}, \qquad |\mathrm{cacc}| \le 128 \cdot 127^2 < 2^{21}
$$

$\mathrm{cacc}$ vừa **22 bit có dấu**. Giá trị thực:

$$
\mathrm{ctx}_{id} = \sum_j p_{ij} v_{jd} \approx \mathrm{cacc}_{id} \cdot m^V_d\, 2^{-n^V_d} \cdot \frac{2^{15}}{127 \cdot E_i}
$$

$1/(127 E_i)$ được tính **một lần cho mỗi hàng** bằng `reciprocal_fixed` (mục 12):
$\frac{1}{127 E_i} \approx y_i \cdot 2^{-s_i}$. Do đó:

$$
\boxed{\ [\mathrm{ctx}_{id}] = \operatorname{rshift}\!\Big(\mathrm{cacc}_{id} \cdot m^V_d \cdot y_i,\ \ n^V_d + s_i - 15 - 16\Big)\ }
$$

Tích lớn nhất là $22 + 7 + 16 = 45$ bit. 12 head được ghép lại thành vector 384 chiều ở Q.16.

---

## 7. Cộng residual

Cả hai toán hạng đều là Q.16, nên:

$$
[h'] = [\mathrm{dense}(\cdot)] + [h]
$$

Đây là phép cộng chính xác, không cần căn chỉnh scale, và kết quả đi thẳng vào LayerNorm.

---

## 8. LayerNorm

### Công thức gốc

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i,
\qquad
\mu = \frac{1}{N}\sum_{j} x_j,
\qquad
\sigma^2 = \frac{1}{N}\sum_{j} (x_j - \mu)^2
$$

### Bỏ phép chia cho $N$

Đặt $S = \sum_j x_j$ và $d_i = N x_i - S = N (x_i - \mu)$. Khi đó

$$
x_i - \mu = \frac{d_i}{N}, \qquad
\sigma^2 = \frac{1}{N} \sum_{j} \left(\frac{d_j}{N}\right)^2 = \frac{\lVert d \rVert^2}{N^3}
\quad\Rightarrow\quad
\frac{x_i - \mu}{\sigma} = \sqrt{N} \cdot \frac{d_i}{\lVert d \rVert}
$$

$$
\boxed{\ y_i = \left(\gamma_i \sqrt{N}\right) \cdot \frac{d_i}{\lVert d \rVert} + \beta_i\ }
$$

- $N x_i = (x_i \ll 8) + (x_i \ll 7)$, vì $384 = 256 + 128$.
- $\gamma_i \sqrt{N}$ là hằng số, tính sẵn offline.
- $|d_i| / \lVert d \rVert \le 1$, nên biểu diễn gọn được trong 18 bit.
- **Scale đầu vào bị triệt tiêu**: nhân mọi $x_i$ với cùng $c > 0$ thì kết quả không đổi.
- $\varepsilon = 10^{-12}$ được bỏ qua.

### Hằng số offline

$$
G_i = \operatorname{round}\!\left(\gamma_i \sqrt{384} \cdot 2^{10}\right) \ (\mathrm{Q}.10), \qquad
B_i = \operatorname{round}\!\left(\beta_i \cdot 2^{16}\right) \ (\mathrm{Q}.16)
$$

Trên cả 25 LayerNorm, $\max|\gamma| = 3.43$, nên
$\max |G| = 68\,767 < 2^{17}$, vừa **18 bit có dấu**.

### Các bước trên FPGA

**Bước 1: tổng và độ lệch.**

$$
S = \sum_{j} [x_j], \qquad d_i = N [x_i] - S
$$

**Bước 2: dịch động** để bình phương vừa cổng 18 bit của DSP:

$$
k = \max\!\left(\operatorname{bitlen}\!\left(\max_i |d_i|\right) + 1 - 18,\ 0\right), \qquad e_i = d_i \gg k
$$

**Bước 3: tổng bình phương.**

$$
Q = \sum_i e_i^2 < 384 \cdot 2^{34} < 2^{43}
$$

$Q$ vừa accumulator 48 bit.

**Bước 4:** $1/\sqrt{Q} \approx y \cdot 2^{-s}$, xem mục 12.

**Bước 5: chuẩn hoá về Q.17.**

$$
n_i = \operatorname{sat}_{18}\!\Big(\operatorname{rshift}(e_i \cdot y,\ s - 17)\Big)
$$

Nếu $Q = 0$ thì đặt $n_i = 0$.

**Bước 6: nhân gamma, cộng beta.**

$$
\boxed{\ [y_i] = \operatorname{rshift}\!\left(n_i \cdot G_i,\ 17 + 10 - 16\right) + B_i\ }
$$

Kết quả ở Q.16, vừa 24 bit có dấu.

---

## 9. GELU

### Công thức gốc

$$
\operatorname{GELU}(x) = x \cdot \Phi(x), \qquad \Phi(x) = \tfrac{1}{2}\left(1 + \operatorname{erf}(x / \sqrt{2})\right)
$$

### LUT cho $\Phi$

$\Phi$ trơn và bị chặn trong $[0, 1]$, nên chỉ cần tra bảng $\Phi$ rồi nhân với $x$. Bảng
phủ $[-8, 8)$ với bước $2^{-6}$, tổng cộng 1024 ô ở Q.15 (vừa một BRAM18):

$$
\mathrm{PHI}[i] = \operatorname{round}\!\left(2^{15}\, \Phi\!\left(\frac{i - 512}{64}\right)\right), \quad i = 0..1023
$$

GELU nhận **trực tiếp đầu ra Q.16 của `dense1`**, chưa làm tròn về INT8:

$$
\mathrm{idx} = \operatorname{clip}\!\Big(\operatorname{rshift}([x],\ 16 - 6) + 512,\ 0,\ 1023\Big)
$$

$$
\boxed{\ [\operatorname{GELU}(x)] = \operatorname{rshift}\!\left([x] \cdot \mathrm{PHI}[\mathrm{idx}],\ 15\right)\ }
$$

Ngoài khoảng $[-8, 8)$ thì $\Phi \approx 0$ hoặc $1$, nên chỉ cần bão hoà chỉ số là cho đúng kết quả.
Đo được: **sai số tuyệt đối lớn nhất $1.9 \times 10^{-3}$** trên $[-12, 12]$.

---

## 10. Mean pooling

$$
\bar{h}_d = \frac{1}{c} \sum_{i:\,M_i = 1} h_{id}, \qquad c = \sum_i M_i
$$

Cộng dồn số nguyên, rồi nhân với nghịch đảo của số token (mục 12):

$$
\Sigma_d = \sum_{i} M_i [h_{id}], \qquad \frac{1}{c} \approx y \cdot 2^{-s}, \qquad
\boxed{\ [\bar{h}_d] = \operatorname{rshift}(\Sigma_d \cdot y,\ s)\ }
$$

Đầu ra là sentence embedding Q.16 INT32, kích thước $(B, 384)$.

---

## 11. Score RAG: cosine

$$
\operatorname{score}(a, b) = \frac{a \cdot b}{\lVert a \rVert\, \lVert b \rVert}
$$

**Bước 1: dịch động** mỗi vector về 16 bit, như bước 2 của LayerNorm. Scale của mỗi vector
bị triệt tiêu trong cosine, nên phép dịch không làm đổi kết quả.

$$
\hat{a} = a \gg k_a, \qquad \hat{b} = b \gg k_b
$$

**Bước 2: tích vô hướng và chuẩn.**

$$
\mathrm{dot} = \sum \hat{a}_d \hat{b}_d \ (< 2^{39}), \qquad
\frac{1}{\lVert \hat{a} \rVert} \approx y_a 2^{-s_a}, \qquad
\frac{1}{\lVert \hat{b} \rVert} \approx y_b 2^{-s_b}
$$

**Bước 3:** chia cho $\lVert \hat{a} \rVert$ trước, để kết quả trung gian không vượt 64 bit:

$$
u = \operatorname{rshift}(\mathrm{dot} \cdot y_a,\ s_a - 15), \qquad
\boxed{\ \operatorname{score} = \operatorname{rshift}(u \cdot y_b,\ s_b)\ } \quad (\mathrm{Q}.15)
$$

Trong hệ thống RAG, vector tài liệu có thể được chuẩn hoá sẵn khi tạo chỉ mục. Khi đó lúc
truy vấn chỉ còn tích vô hướng và một lần rsqrt cho câu truy vấn.

---

## 12. Căn bậc 2 nghịch đảo và nghịch đảo

Cả hai dùng cùng một cách: **chuẩn hoá, tra LUT, một bước Newton**.

### 12.1. rsqrt (`utils/rsqrt.py`)

Mục tiêu: $1/\sqrt{v} \approx y \cdot 2^{-s}$ với $v$ tới khoảng $2^{47}$.

**Chuẩn hoá về $[1, 4)$** với số mũ chẵn:

$$
E = (\operatorname{bitlen}(v) - 1) \mathbin{\&} \lnot 1, \qquad
m_{16} = v \cdot 2^{16 - E} \in [2^{16}, 2^{18})
$$

**Tra LUT** (768 ô × 16 bit, lấy tại tâm ô):

$$
\mathrm{idx} = (m_{16} \gg 8) - 256, \qquad
y_0 = \mathrm{RSQRT}[\mathrm{idx}] = \operatorname{round}\!\left(\frac{2^{16}}{\sqrt{1 + (\mathrm{idx} + 0.5)/256}}\right)
$$

**Newton** cho $f(y) = y^{-2} - m$, tức $y_1 = y_0 (3 - m y_0^2) / 2$:

$$
u = (y_0^2) \gg 16, \qquad t = (m_{16}\, u) \gg 16, \qquad y_1 = \big(y_0 (3 \cdot 2^{16} - t)\big) \gg 17
$$

$$
\boxed{\ \frac{1}{\sqrt{v}} \approx y_1 \cdot 2^{-(16 + E/2)}\ }
$$

Sai số tương đối lớn nhất đo được: $3.6 \times 10^{-5}$.

**Ví dụ** với $v = 1000$:

| Bước | Giá trị |
|---|---|
| $E$ | $8$ |
| $m_{16}$ | $256\,000$ ($m = 3.906$) |
| $\mathrm{idx}$ | $744$ |
| $y_0$ | $33\,151$ |
| $y_1$ | $33\,159$ |
| $y_1 \cdot 2^{-20}$ | $0.0316229$ |
| $1/\sqrt{1000}$ (chính xác) | $0.0316228$ |

### 12.2. Nghịch đảo (`utils/reciprocal.py`)

Mục tiêu: $1/v \approx y \cdot 2^{-s}$. Dùng ở softmax ($v = 127 E_i$) và mean pooling ($v = c$).

**Chuẩn hoá về $[1, 2)$:**

$$
p = \operatorname{bitlen}(v) - 1, \qquad m_{16} = v \cdot 2^{16 - p} \in [2^{16}, 2^{17})
$$

**Tra LUT** (256 ô × 16 bit):

$$
\mathrm{idx} = (m_{16} \gg 8) - 256, \qquad
y_0 = \mathrm{RECIP}[\mathrm{idx}] = \operatorname{round}\!\left(\frac{2^{16}}{1 + (\mathrm{idx} + 0.5)/256}\right)
$$

**Newton** cho $f(y) = 1/y - m$, tức $y_1 = y_0 (2 - m y_0)$:

$$
t = (m_{16}\, y_0) \gg 16, \qquad y_1 = \big(y_0 (2 \cdot 2^{16} - t)\big) \gg 16
$$

$$
\boxed{\ \frac{1}{v} \approx y_1 \cdot 2^{-(16 + p)}\ }
$$

Sai số tương đối lớn nhất đo được: $2.9 \times 10^{-5}$.

---

## 13. Bảng hằng số, LUT và độ rộng bit

### Hằng số

| Tên | Giá trị | Ý nghĩa |
|---|---|---|
| $R_{127}$ | $\lceil 2^{24}/127 \rceil = 132\,105$ | chia cho 127 khi tính scale |
| $C$ | $\operatorname{round}(2^{15}/\sqrt{32}) = 5\,793$ | $1/\sqrt{D}$ trong điểm attention |
| $\mathrm{LOG2E}$ | $\operatorname{round}(2^{15} \log_2 e) = 47\,274$ | đổi $e^{-t}$ sang $2^{-w}$ |
| $G_i$ | Q.10, 18 bit | $\gamma_i \sqrt{384}$, mỗi LayerNorm 384 giá trị |
| $B_i$, $[b_j]$ | Q.16 | beta của LayerNorm, bias của `Linear` |

### LUT

| LUT | Số ô | Giá trị | Định dạng |
|---|---|---|---|
| $\mathrm{RECIP}[m]$ (lượng tử hoá) | 128 | $2^{22}/m$ | 23 bit không dấu |
| $\mathrm{EXP2}$ (softmax) | 257 | $[16\,384,\ 32\,768]$ | Q.15 |
| $\mathrm{PHI}$ (GELU) | 1024 | $[0,\ 32\,768]$ | Q.15 |
| $\mathrm{RSQRT}$ | 768 | $[32\,776,\ 65\,472]$ | Q.16 |
| $\mathrm{RECIP}$ (nghịch đảo) | 256 | $[32\,800,\ 65\,408]$ | Q.16 |

### Bộ nhớ weight

| Dữ liệu | Kích thước |
|---|---|
| Bảng word embedding | $250\,037 \times 384$ INT8, cộng 12 bit scale cho mỗi hàng |
| Weight `Linear` của 12 layer | $12 \times (4 \cdot 384^2 + 2 \cdot 384 \cdot 1536)$ INT8 ≈ 21 MB, cộng scale cho mỗi cột |

### Các accumulator

| Accumulator | Độ rộng | Mục |
|---|---|---|
| `Linear` $\mathrm{acc}$ | $\le 26$ bit | 5 |
| Điểm attention $\mathrm{sacc}$ | 20 bit | 6.2 |
| Tổng softmax $E$ | 23 bit | 6.3 |
| Context $\mathrm{cacc}$ | 22 bit | 6.4 |
| LayerNorm $Q = \sum e^2$ | 43 bit | 8 |

Mọi phép nhân đều có toán hạng $\le 27 \times 18$ bit, vừa DSP48E2. Các tích rộng hơn
(tới 47 bit) được ghép từ hai phép nhân theo kiểu cascade.

---

## 14. Độ chính xác

### Từng khối, so với float64

| Khối | Sai số |
|---|---|
| rsqrt | $3.6 \times 10^{-5}$ tương đối |
| Nghịch đảo | $2.9 \times 10^{-5}$ tương đối |
| $e^{-t}$ (softmax) | $1.3 \times 10^{-3}$ tuyệt đối |
| GELU | $1.9 \times 10^{-3}$ tuyệt đối |
| LayerNorm (25 LN, có outlier) | $9.5 \times 10^{-4}$ lớn nhất, $7.9 \times 10^{-5}$ trung bình |

### Cả pipeline so với PyTorch float32

Đo bằng `python model/main.py`, với sai số phần trăm

$$
e_d = 100 \cdot \frac{\left|\hat{y}_d - y_d\right|}{\max y - \min y}\ (\%)
$$

trong đó $\hat{y}$ là kết quả NumPy (đổi Q.16 sang số thực) và $y$ là kết quả PyTorch.

| Câu | Sai số TB | Sai số lớn nhất | Cosine |
|---|---|---|---|
| Tôi đang làm đồ án FPGA. | 0.45% | 2.48% | 0.99918 |
| Đồ án dùng bo mạch KV260 để tăng tốc mô hình nhúng câu. | 0.59% | 2.06% | 0.99915 |
| AXI hỗ trợ nhiều giao dịch outstanding nhờ ID của từng giao dịch. | 0.64% | 3.49% | 0.99885 |
| Hôm nay trời mưa nên tôi ở nhà đọc sách. | 0.40% | 1.84% | 0.99958 |
| Systolic array thực hiện phép nhân ma trận INT8 trên FPGA. | 0.72% | 2.32% | 0.99868 |

Score RAG với query "Tôi đang làm đồ án FPGA.":

| Passage | PyTorch | NumPy | Sai số | Hạng PT | Hạng NP |
|---|---|---|---|---|---|
| Systolic array … | 0.46591 | 0.46008 | 0.0058 | 1 | 1 |
| Đồ án dùng bo mạch KV260 … | 0.45370 | 0.45987 | 0.0062 | 2 | 2 |
| AXI hỗ trợ … | 0.20064 | 0.19800 | 0.0026 | 3 | 3 |
| Hôm nay trời mưa … | 0.13447 | 0.12961 | 0.0049 | 4 | 4 |

Trên bộ 6 câu thử khác, sai số score lớn nhất là 0.013. Có một lần hai tài liệu không liên
quan đổi chỗ cho nhau (score 0.053 và 0.041), còn các kết quả đứng đầu luôn giữ đúng thứ tự.

So với thiết kế INT8 per-tensor cũ (cosine 0.847–0.914, sai số TB 7.1%), sai số giảm
khoảng 10 lần.
