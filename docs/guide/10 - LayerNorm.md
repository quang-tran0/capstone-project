---
tags: [minilm, fpga, layernorm]
---

# Chapter 10 — LayerNorm

← [[09 - Self-attention and softmax]] | Next: [[11 - GELU and residual]] →

Code: `model/numpy/layers/int_layer_norm.py`

LayerNorm appears 25 times in MiniLM (1 after the embedding, 2 in each encoder layer). The
original formula needs a division and a square root, neither of which exists on the integer
datapath. This chapter shows how to remove both.

## 10.1. Original formula

For one row of $N = 384$ elements:

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i,
\qquad
\mu = \frac{1}{N}\sum_{j} x_j,
\qquad
\sigma^2 = \frac{1}{N}\sum_{j} (x_j - \mu)^2
$$

## 10.2. Key idea: remove the division by $N$

Let

$$
d_i = N x_i - \sum_j x_j = N (x_i - \mu)
$$

$d$ is computed exactly with integers (no division). Then:

- $\lVert d \rVert = N \sqrt{\sum (x - \mu)^2}$
- $\sigma = \sqrt{\sum (x - \mu)^2 / N}$

Substituting, the factors of $N$ cancel and only $\sqrt{N}$ remains:

$$
\frac{x_i - \mu}{\sigma} = \sqrt{N} \cdot \frac{d_i}{\lVert d \rVert}
\qquad\Rightarrow\qquad
y_i = \underbrace{\left(\gamma_i \sqrt{N}\right)}_{G_i} \cdot \frac{d_i}{\lVert d \rVert} + \beta_i
$$

Two things make the hardware simple:

1. **$\sqrt{N}$ is a constant**, folded into $\gamma$ offline: $G = \gamma \cdot \sqrt{384}$,
   stored in Q.10, 18 bits.
2. **$d / \lVert d \rVert$ does not depend on the input scale**: multiplying the whole row by a
   constant leaves the ratio unchanged. So the whole row can be shifted by any number of bits
   without error.

$\varepsilon = 10^{-12}$ is ignored because it is negligible compared with the actual variance.

The rest (steps 3–7 below) does one thing: **compute $d / \lVert d \rVert$ with integers**, then
multiply by $G$ and add $\beta$.

## 10.3. The steps, with real data

Example: the LayerNorm after the embedding, 2nd token of the sentence "Xin chào". Only the first
4 of 384 elements are shown.

### Steps 1–2: compute $d$

```
x[:4] = 796, -2268, -66, -814          (Q.16 → 0.01215, -0.03461, -0.00101, -0.01242)
d = 384·x − Σx
d[:4] = 327657, -848919, -3351, -290583
```

$384 \cdot x$ can be done with shifts and an add: $384 = 256 + 128$, so $384x = (x \ll 8) + (x \ll 7)$.

### Step 3: shrink $d$ to fit 18 bits

**Problem:** $d$ needs up to 44 bits (values around 1.9 million). The next step squares $d$. A
direct 44 × 44-bit multiply gives 88 bits, which no DSP can hold. DSP48E2 only multiplies
$27 \times 18$ bits.

**Solution:** shift **the whole row** right by the same $s$ bits so that the largest element just
fits in 18 signed bits:

```
max|d|            = 1 936 023
leading_one(...)  = 21            (this value needs 21 bits)
s = 21 + 1 − 18   = 4             (+1 for the sign; to keep 18 bits, drop 4 low bits)
e = d >> 4
e[:4] = 20478, -53058, -210, -18162        max|e| = 121002 < 2¹⁷ ✓
```

**Why the result is still correct:** shifting right by 4 divides every element by 16, so
$\lVert d \rVert$ is also divided by 16:

$$
\frac{e}{\lVert e \rVert} = \frac{d/16}{\lVert d \rVert / 16} = \frac{d}{\lVert d \rVert}
$$

Only the 4 low bits of each element are lost. The largest element still keeps all 18 bits, so the
loss is tiny (relative precision about $2^{-17}$). Each row has its own $s$.

### Step 4: sum of squares

```
S = Σ e²  (384 values)  = 718 860 868 128          (needs 41 bits)
```

Each $e^2$ is an $18 \times 18$-bit multiply (at most 36 bits). Summing 384 of them adds about
9 bits, at most 45 bits, which fits the **48-bit accumulator** of a DSP48E2. Now
$\lVert e \rVert = \sqrt{S}$.

### Step 5 - inverse square root of S

No square root and no division are used.

Target: $1/\sqrt{718\,860\,868\,128} \approx 1.17944 \times 10^{-6}$. Three techniques (details in
[[13 - rsqrt and reciprocal]]):

**(a) Split $S$ into mantissa × power of two**, like floating point:

```
S has 40 bits  →  exponent E = 38   (chosen even)
S = m · 2³⁸    with m = 2.615       (always in [1, 4))
1/√S = 1/√m · 2⁻¹⁹
```

The exponent must be even so that $38/2 = 19$ is an integer; then the power-of-two part is just a
shift. The hard part shrinks to $1/\sqrt{m}$ with $m$ in the small range $[1, 4)$.

**(b) Table lookup** of $1/\sqrt{m}$: 768 entries splitting $[1, 4)$ into steps of $1/256$. Gives
$y \approx 0.6184$.

**(c) One Newton step** to refine: $y \leftarrow y \cdot (3 - m y^2) / 2$. The relative error drops
from about $10^{-3}$ to below $10^{-4}$.

**The result** is a `Scale(m, n)`:

```
1/√S ≈ 40525 · 2⁻³⁵ = 1.1794327e-6        (exact: 1.1794447e-6)
```

($0.6184 \approx 40525 \cdot 2^{-16}$, plus the exponent part $2^{-19}$, gives $2^{-35}$ in total.)

This step runs **once per row**, so in hardware it can be sequential without affecting throughput.

### Step 6: compute $d / \lVert d \rVert = e \cdot (1/\sqrt{S})$

Each element is multiplied by $m = 40525$ and shifted so the result is in **Q.17**:

```
norm = (e · 40525) >> (35 − 17) = (e · 40525) >> 18

e[0] = 20478  →  20478·40525 >> 18 = 3166   →  3166 / 2¹⁷ = 0.024155

norm[:4]            = 3166, -8202, -32, -2808  →  0.024155, -0.062576, -0.000244, -0.021423
exact d/‖d‖ [:4]                                →  0.024153, -0.062578, -0.000247, -0.021420
```

The shift is $35 - 17 = 18$ bits because the real value is $e \cdot 40525 \cdot 2^{-35}$, and
storing it in Q.17 (an extra factor $2^{17}$) leaves a right shift of 18 bits.

**Why Q.17:** $d / \lVert d \rVert$ always lies in $[-1, 1]$ (no element is larger than the length
of the whole vector). Q.17 plus the sign bit is exactly 18 bits, matching the 18-bit DSP port of
step 7.

**Saturation at $\pm(2^{17} - 1)$:** a value of exactly 1.0 only happens when the row has a single
non-zero element. $2^{17}$ does not fit in 18 signed bits, so it is clamped to $2^{17} - 1$.

### Step 7: multiply by $G$, add $\beta$

$G = \gamma \sqrt{384}$ is precomputed in Q.10:

```
γ[:4] = 0.8518, 0.8578, 0.7717, 0.7678
G = round(γ · √384 · 2¹⁰) = 17092, 17213, 15485, 15406
```

$$
y = (\text{norm} \cdot G) \gg 11 + \beta
$$

The shift is 11 bits because norm has 17 fraction bits and $G$ has 10, giving 27; Q.16 needs
$27 - 16 = 11$ of them dropped. $\beta$ is already Q.16 and is added directly.

```
y[0] = (3166 · 17092) >> 11 + (−2012) = 26422 − 2012 = 24410  →  0.372467

y (NumPy) [:4]  = 0.372467, -1.158478, 0.026840, -0.394821
float LayerNorm = 0.372452, -1.158512, 0.026797, -0.394778
```

The difference is about $10^{-5}$ to $4 \times 10^{-5}$, a few LSBs of Q.16.

### Special case

Constant row ($d = 0$, $S = 0$): rsqrt is skipped and the result is $\beta$.

## 10.4. Summary

| Step | What | Width | Why |
|---|---|---|---|
| 1–2 | $d = 384x - \sum x$ | ~44 bits | Exact centering, no division |
| 3 | $e = d \gg s$ | 18 bits | Fits the 18-bit multiplier port; $d/\lVert d\rVert$ unchanged |
| 4 | $S = \sum e^2$ | 48 bits | $\lVert e \rVert^2$ in the DSP accumulator |
| 5 | $1/\sqrt{S}$ = table + Newton + exponent shift | 17-bit mantissa | No sqrt, no division; once per row |
| 6 | $\text{norm} = e \cdot m \gg (n - 17)$ | 18 bits, Q.17 | Gives $d/\lVert d \rVert \in [-1, 1]$ |
| 7 | $y = \text{norm} \cdot G \gg 11 + \beta$ | 32 bits, Q.16 | $\sqrt{N}$ already folded into $G$ |

## 10.5. On the FPGA

- **Only 3 multiplies per element**: $e^2$ ($18 \times 18$), $e \cdot m$ ($18 \times 17$),
  $\text{norm} \cdot G$ ($18 \times 18$). Each fits one DSP48E2.
- **The whole row must be buffered** ($384 \times 33$ bits $\approx$ 1.6 KB) in BRAM, because the
  row is scanned several times: compute $\sum x$; compute $d$ and $\max |d|$; compute $\sum e^2$;
  finally normalize each element.
- rsqrt runs once per row, so it can be sequential and small.

## 10.6. Accuracy

- Tested with random $\gamma$, $\beta$ and inputs with outlier channels 30 times larger: largest
  error versus float LayerNorm below $2 \times 10^{-3}$, mean below $2 \times 10^{-4}$.
- In the comparison report, the LayerNorm rows show a mean error of about 0.03–0.06%, much lower
  than the surrounding steps. LayerNorm also **pulls the accumulated error back down** after each block.

---

← [[09 - Self-attention and softmax]] | Next: [[11 - GELU and residual]] →
