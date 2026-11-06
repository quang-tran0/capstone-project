---
tags: [minilm, fpga, attention, softmax]
---

# Chapter 9 — Self-attention and softmax

← [[08 - Embedding]] | Next: [[10 - LayerNorm]] →

Code: `model/numpy/layers/self_attention.py`, `model/numpy/utils/softmax.py`

## 9.1. Original formula

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{D}} + M\right) V
$$

$M$ is the mask: $-\infty$ at padding tokens so they receive no attention. MiniLM has 12 heads,
each doing this over 32 dimensions.

Three parts are hard with integers: multiplying by $1/\sqrt{32}$, the `exp` in softmax, and the
division by the sum in softmax. This chapter handles them in turn.

## 9.2. Q, K, V

The input $[h]$ (Q.16) is quantized **once** per token and shared by the three `Linear`s
([[07 - Linear]]). The results $[Q], [K], [V]$ in Q.16 are split into 12 heads and **quantized
again**:

- $Q$, $K$: one scale per token (within each head).
- $V$: one scale per channel (within each head), because the context step sums over tokens
  (see [[06 - Quantization#6.4. Which axis a scale covers]]).

## 9.3. Attention scores: folding $1/\sqrt{D}$ into the scale

$$
\mathrm{sacc}_{ij} = \sum_{d=1}^{32} Q_{q,id}\, K_{q,jd}, \qquad |\mathrm{sacc}| \le 32 \cdot 127^2 < 2^{19}
$$

A signed 20-bit register is enough. The real score is

$$
z_{ij} = \mathrm{sacc}_{ij} \cdot S^Q_i \cdot S^K_j \cdot \frac{1}{\sqrt{32}}
$$

The constant $1/\sqrt{32}$ is also written as $m \cdot 2^{-n}$: `Scale.const(5793, 15)`, since
$5793 / 2^{15} = 0.17679 \approx 1/\sqrt{32}$. So all three factors are just a product of three scales:

```python
score_scale = q.scale * k.scale.swapaxes(-2, -1) * INV_SQRT_HEAD_DIM
scores = score_scale.rescale(score_accumulator, 16)        # Q.16
```

$$
[z_{ij}] = \operatorname{round\_shift}\!\Big(\mathrm{sacc}_{ij} \cdot m^Q_i\, m^K_j \cdot 5793,\ \ n^Q_i + n^K_j + 15 - 16\Big)
$$

The widest product is $20 + 7 + 7 + 13 = 47$ bits.

## 9.4. Integer softmax

$$
p_{ij} = \frac{e^{z_{ij}}}{\sum_{j'} e^{z_{ij'}}} = \frac{e^{z_{ij} - z_{\max}}}{\sum_{j'} e^{z_{ij'} - z_{\max}}}
$$

### Step 1 — Subtract the max

$$
[t_{ij}] = \max_{j'} [z_{ij'}] - [z_{ij}] \ge 0 \qquad \text{(only tokens with mask = 1)}
$$

After this step every exponent is $\le 0$, so $e^{-t} \in (0, 1]$ and never overflows.

### Step 2 — Switch from base $e$ to base 2

Since $e^{-t} = 2^{-t \log_2 e}$, multiply by the constant
$\mathrm{LOG2E} = \operatorname{round}(2^{15} \log_2 e) = 47\,274$ (Q.15):

$$
[w] = \operatorname{round\_shift}\!\left([t] \cdot \mathrm{LOG2E},\ 15\right) \qquad \text{(Q.16)}
$$

Why base 2: a power of two with an integer exponent is just a shift.

### Step 3 — Split integer and fractional parts

$$
w_I = [w] \gg 16, \qquad w_F = [w] \mathbin{\&} (2^{16} - 1), \qquad
2^{-w} = 2^{-w_F / 2^{16}} \cdot 2^{-w_I}
$$

- $2^{-w_I}$: shift right by $w_I$ bits.
- $2^{-w_F/2^{16}}$ with the fractional part in $[0, 1)$: table lookup.

### Step 4 — Look up and shift

A 257-entry table in Q.15 covering $[0, 1]$ with step $1/256$:

$$
\mathrm{EXP2}[k] = \operatorname{round}\!\left(2^{15} \cdot 2^{-k/256}\right), \quad k = 0..256
$$

$$
\varepsilon_{ij} = \begin{cases}
\mathrm{EXP2}\big[\operatorname{round\_shift}(w_F, 8)\big] \gg w_I & w_I \le 15 \text{ and mask} = 1 \\
0 & \text{otherwise}
\end{cases}
$$

The largest element has $t = 0$, so $\varepsilon = 2^{15}$ (i.e. 1.0). If $w_I > 15$ then
$e^{-t} < 2^{-15}$, too small, treated as 0.

### Step 5 — Sum and probability codes

$$
E_i = \sum_j \varepsilon_{ij} \ \ (\text{Q.15}), \qquad
P_{q,ij} = \operatorname{round\_shift}\!\left(127 \cdot \varepsilon_{ij},\ 15\right) \in [0, 127]
$$

$P_q$ is an INT8 code: the largest element maps to 127. Then

$$
p_{ij} = \frac{\varepsilon_{ij}}{E_i} \approx \frac{P_{q,ij}}{127 \cdot E_i}
$$

**Division by $E_i$ is not done per element**; it is folded into the context step below. The whole
row needs a single reciprocal.

## 9.5. Context

$$
\mathrm{cacc}_{id} = \sum_{j} P_{q,ij}\, V_{q,jd}, \qquad |\mathrm{cacc}| \le 128 \cdot 127^2 < 2^{21}
$$

A signed 22-bit register. The real value:

$$
\mathrm{ctx}_{id} = \sum_j p_{ij} v_{jd} \approx \mathrm{cacc}_{id} \cdot S^V_d \cdot \frac{1}{127 \cdot E_i}
$$

$1/(127 E_i)$ is computed **once per row** by `reciprocal_fixed` ([[13 - rsqrt and reciprocal]]),
which returns a `Scale`. Multiply it with the scale of $V$ and rescale:

```python
inverse_total = reciprocal_fixed(127 * total, self.reciprocal_lut)
context = (v.scale * inverse_total).rescale(context_accumulator, 16)
```

The widest product is $22 + 7 + 17 = 46$ bits. The 12 heads are concatenated back into a
384-dimensional vector and stored in a 32-bit Q.16 register.

## 9.6. Register summary

| Signal | Width |
|---|---|
| $Q_q$, $K_q$, $V_q$ | 8 bits |
| $\mathrm{sacc}$ | 20 bits |
| $[z]$ (scores) | Q.16 |
| $\varepsilon$ | Q.15, 16 bits |
| $E$ (sum) | $\le 2^{22}$ for 128 tokens |
| $P_q$ | 8 bits, $[0, 127]$ |
| $\mathrm{cacc}$ | 22 bits |
| context | 32 bits, Q.16 |

---

← [[08 - Embedding]] | Next: [[10 - LayerNorm]] →
