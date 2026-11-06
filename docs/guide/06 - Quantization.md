---
tags: [minilm, fpga, quantization]
---

# Chapter 6 — Quantization

← [[05 - Number storage]] | Next: [[07 - Linear]] →

Code: `model/numpy/utils/int_quantize.py`, `model/numpy/hwtypes/scale.py`

**Quantization** converts a tensor into INT8 codes plus scales. This chapter answers three
questions: how to choose a scale, how to compute the codes, and which elements share a scale.

## 6.1. General principle

Quantization is **symmetric**: the scale is chosen so that the element with the largest absolute
value (the **peak**) maps to code 127:

$$
S \approx \frac{\text{peak}}{127}, \qquad x_q = \operatorname{round}\!\left(\frac{x}{S}\right) \in [-127, 127]
$$

Since $S$ must have the form $m \cdot 2^{-n}$, we choose the largest $n$ such that $m$ still fits
in 7 bits, and **round $m$ up**. Rounding up guarantees $S \ge \text{peak}/127$, so the largest
element gets a code $\le 127$: nothing is clipped, and the error is always $\le$ ½ step.

## 6.2. Weights (offline)

Function `quantize_weight`. Floats are allowed because it runs once on a PC.

For each column $j$ of the weight:

$$
s = \frac{\max |W_{:,j}|}{127}, \qquad
n_j = \max\left\{ n \le 31 \;\middle|\; \lceil s \cdot 2^{n} \rceil \le 127 \right\}, \qquad
m_j = \left\lceil s \cdot 2^{n_j} \right\rceil
$$

$$
W_{q,kj} = \operatorname{clip}\!\left(\operatorname{rint}\!\left(\frac{W_{kj}}{m_j 2^{-n_j}}\right), -127, 127\right)
$$

The embedding tables are handled the same way, but per row.

## 6.3. Activations (at run time)

Function `quantize`. Runs on the FPGA for every token, so **integers only**. The input is one row
$x$ in Q.16.

**Step 1 — Find the peak:** $P = \max_k |x_k|$ (comparator tree).

**Step 2 — Divide by 127 with a multiplication.** There is no divider, so multiply by the constant
$R = \lceil 2^{24} / 127 \rceil = 132\,105$:

$$
P \cdot R \approx \frac{P}{127} \cdot 2^{24}
$$

**Step 3 — Choose $n$.** For every $n$ from 0 to 31, compute a candidate $m(n)$ with a
round-up right shift:

$$
m(n) = \left\lceil P \cdot R \cdot 2^{n - 40} \right\rceil
$$

($40 = 16 + 24$: the 16 fraction bits of Q.16 and the 24 bits of $R$.) Take the largest $n$ with
$m(n) \le 127$. In hardware, the 32 shifts run in parallel followed by a **priority encoder**.
If $P = 0$, set $m = 1$, $n = 31$.

**Step 4 — Compute the codes without division.** We need $x / S = x \cdot 2^{n} / m$. Division by
$m$ is replaced by a lookup in a 128-entry reciprocal ROM:

$$
\mathrm{RECIP}[m] = \operatorname{round}\!\left(\frac{2^{22}}{m}\right), \qquad
x_{q,k} = \operatorname{clip}\!\Big(\operatorname{round\_shift}\!\left(x_k \cdot \mathrm{RECIP}[m],\ 22 + 16 - n\right),\ -127,\ 127\Big)
$$

### Example (2nd token of "Xin chào FPGA", before $W_Q$ in layer 0)

| Quantity | Value |
| --- | --- |
| Peak $P$ | 2.3419 |
| $P / 127$ | 0.018440 |
| Chosen scale | $m = 76$, $n = 12$: $76 \cdot 2^{-12} = 0.018555$ (≥ 0.018440 because of rounding up) |
| $x$ (real) | 0.3725, −1.1585, 0.0268, −0.3948, −1.0497, −0.8299 |
| INT8 codes | 20, −62, 1, −21, −57, −45 |

Check: $20 \times 0.018555 = 0.371 \approx 0.3725$.

## 6.4. Which axis a scale covers

| Tensor | One scale per | Shape of $(m, n)$ |
|---|---|---|
| `Linear` weight | column (output channel) | $(1, \text{out})$ |
| Embedding table | row (token ID) | $(V, 1)$ |
| `Linear` input | token | $(B, L, 1)$ |
| $Q$, $K$ in attention | token × head | $(B, H, L, 1)$ |
| $V$ in attention | channel × head | $(B, H, 1, D)$ |

**Rule:** a scale must lie **outside the sum** of the matrix multiply. For $y = XW$ the sum runs
over $k$:

$$
y_{ij} = \sum_k x_{ik} W_{kj} \approx S_{x,i} S_{w,j} \sum_k x_{q,ik} W_{q,kj}
$$

A scale may depend on row $i$ of $X$ and column $j$ of $W$, but not on $k$. For the context $PV$
in attention, the sum runs over tokens, so $V$ needs per-channel scales rather than per-token ones.

## 6.5. Why not one scale per tensor

Transformer activations have a few **outlier channels** tens of times larger than the rest. With
one scale for the whole tensor, the scale is pulled up by the outliers, and normal values only use
a handful of INT8 levels.

Experiment with simulated quantization inserted into PyTorch (cosine against the float model):

| Option | Smallest cosine | Largest score error |
|---|---|---|
| INT8 with one scale per tensor (old design) | 0.849 | 0.119 |
| Keep `Linear` outputs wide | 0.920 | 0.109 |
| + per-channel weights | 0.958 | 0.052 |
| **+ per-token activations** | **0.998** | **0.010** |

---

← [[05 - Number storage]] | Next: [[07 - Linear]] →
