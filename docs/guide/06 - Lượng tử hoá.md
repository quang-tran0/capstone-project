---
tags: [minilm, fpga, luong-tu-hoa]
---

# Chương 6 — Lượng tử hoá

← [[05 - Cách lưu số]] | Tiếp: [[07 - Linear]] →

Code: `model/numpy/utils/int_quantize.py`, `model/numpy/hwtypes/scale.py`

**Lượng tử hoá** là đổi một tensor sang dạng mã INT8 kèm scale. Chương này trả lời ba câu
hỏi: chọn scale thế nào, tính mã thế nào, và mỗi scale dùng chung cho những phần tử nào.

## 6.1. Nguyên tắc chung

Lượng tử hoá **đối xứng**: scale được chọn sao cho phần tử có giá trị tuyệt đối lớn nhất (gọi
là **đỉnh**, peak) ứng với mã 127:

$$
S \approx \frac{\text{peak}}{127}, \qquad x_q = \operatorname{round}\!\left(\frac{x}{S}\right) \in [-127, 127]
$$

Vì $S$ phải có dạng $m \cdot 2^{-n}$, ta chọn $n$ lớn nhất sao cho $m$ còn vừa 7 bit, và **làm
tròn $m$ lên**. Làm tròn lên bảo đảm $S \ge \text{peak}/127$, nên phần tử lớn nhất cho mã
$\le 127$: không phần tử nào bị cắt, và sai số luôn $\le$ ½ bước.

## 6.2. Weight (offline)

Hàm `quantize_weight`. Được dùng số thực vì chạy một lần trên máy tính.

Với mỗi cột $j$ của weight:

$$
s = \frac{\max |W_{:,j}|}{127}, \qquad
n_j = \max\left\{ n \le 31 \;\middle|\; \lceil s \cdot 2^{n} \rceil \le 127 \right\}, \qquad
m_j = \left\lceil s \cdot 2^{n_j} \right\rceil
$$

$$
W_{q,kj} = \operatorname{clip}\!\left(\operatorname{rint}\!\left(\frac{W_{kj}}{m_j 2^{-n_j}}\right), -127, 127\right)
$$

Bảng embedding làm giống hệt nhưng theo hàng.

## 6.3. Activation (lúc chạy)

Hàm `quantize`. Chạy trên FPGA cho mỗi token, nên **chỉ được dùng số nguyên**. Đầu vào là một
hàng $x$ ở Q.16.

**Bước 1 — Tìm đỉnh:** $P = \max_k |x_k|$ (cây so sánh).

**Bước 2 — Chia cho 127 bằng phép nhân.** Không có bộ chia, nên nhân với hằng số
$R = \lceil 2^{24} / 127 \rceil = 132\,105$:

$$
P \cdot R \approx \frac{P}{127} \cdot 2^{24}
$$

**Bước 3 — Chọn $n$.** Với mỗi $n$ từ 0 đến 31, tính ứng viên $m(n)$ bằng phép dịch phải làm
tròn lên:

$$
m(n) = \left\lceil P \cdot R \cdot 2^{n - 40} \right\rceil
$$

($40 = 16 + 24$: 16 bit thập phân của Q.16 và 24 bit của $R$.) Lấy $n$ lớn nhất mà
$m(n) \le 127$. Trên phần cứng, 32 phép dịch chạy song song rồi qua một **bộ mã hoá ưu tiên**.
Nếu $P = 0$ thì đặt $m = 1$, $n = 31$.

**Bước 4 — Tính mã mà không chia.** Cần $x / S = x \cdot 2^{n} / m$. Phép chia cho $m$ được
thay bằng tra ROM nghịch đảo 128 ô:

$$
\mathrm{RECIP}[m] = \operatorname{round}\!\left(\frac{2^{22}}{m}\right), \qquad
x_{q,k} = \operatorname{clip}\!\Big(\operatorname{round\_shift}\!\left(x_k \cdot \mathrm{RECIP}[m],\ 22 + 16 - n\right),\ -127,\ 127\Big)
$$

### Ví dụ (token thứ 2 của "Xin chào FPGA", trước $W_Q$ ở layer 0)

| Đại lượng | Giá trị |
|---|---|
| Đỉnh $P$ | 2.3419 |
| $P / 127$ | 0.018440 |
| Scale chọn được | $m = 76$, $n = 12$: $76 \cdot 2^{-12} = 0.018555$ (≥ 0.018440 vì làm tròn lên) |
| $x$ (số thực) | 0.3725, −1.1585, 0.0268, −0.3948, −1.0497, −0.8299 |
| Mã INT8 | 20, −62, 1, −21, −57, −45 |

Kiểm tra: $20 \times 0.018555 = 0.371 \approx 0.3725$.

## 6.4. Scale theo chiều nào

| Tensor | Một scale cho mỗi | Shape của $(m, n)$ |
|---|---|---|
| Weight `Linear` | cột (kênh ra) | $(1, \text{out})$ |
| Bảng embedding | hàng (token ID) | $(V, 1)$ |
| Đầu vào `Linear` | token | $(B, L, 1)$ |
| $Q$, $K$ trong attention | token × head | $(B, H, L, 1)$ |
| $V$ trong attention | kênh × head | $(B, H, 1, D)$ |

**Quy tắc:** scale phải **nằm ngoài tổng** của phép nhân ma trận. Với $y = XW$, tổng chạy theo
chỉ số $k$:

$$
y_{ij} = \sum_k x_{ik} W_{kj} \approx S_{x,i} S_{w,j} \sum_k x_{q,ik} W_{q,kj}
$$

Scale được phép phụ thuộc hàng $i$ của $X$ và cột $j$ của $W$, nhưng không được phụ thuộc $k$.
Với context $PV$ trong attention, tổng chạy theo token, nên $V$ phải có scale theo kênh chứ
không theo token.

## 6.5. Vì sao không dùng một scale cho cả tensor

Activation của transformer có vài **kênh outlier** lớn gấp hàng chục lần các kênh khác. Nếu cả
tensor dùng chung một scale, scale bị kéo lên theo outlier, và các giá trị bình thường chỉ còn
dùng được vài mức INT8.

Thí nghiệm chèn lượng tử hoá giả lập vào PyTorch (đo cosine với bản float):

| Phương án | Cosine nhỏ nhất | Sai số score lớn nhất |
|---|---|---|
| INT8 một scale cho cả tensor (thiết kế cũ) | 0.849 | 0.119 |
| Giữ đầu ra `Linear` ở dạng rộng | 0.920 | 0.109 |
| + weight theo kênh | 0.958 | 0.052 |
| **+ activation theo token** | **0.998** | **0.010** |

---

← [[05 - Cách lưu số]] | Tiếp: [[07 - Linear]] →
