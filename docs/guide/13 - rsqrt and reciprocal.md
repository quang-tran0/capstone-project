---
tags: [minilm, fpga, rsqrt, reciprocal]
---

# Chapter 13 — rsqrt and reciprocal

← [[12 - Mean pooling and cosine]] | Next: [[14 - Bit widths and FPGA mapping]] →

Code: `model/numpy/utils/rsqrt.py`, `model/numpy/utils/reciprocal.py`

The model needs two operations the FPGA does not have:

| Operation | Used in |
|---|---|
| $1/\sqrt{v}$ (rsqrt) | LayerNorm ([[10 - LayerNorm]]), cosine ([[12 - Mean pooling and cosine]]) |
| $1/v$ (reciprocal) | softmax: $1/(127E)$ ([[09 - Self-attention and softmax]]), mean pooling: $1/c$ |

Both use the same recipe: **normalize → look up → one Newton step**. The result is a `Scale`
$y \cdot 2^{-s}$ (17-bit mantissa $y$), so it plugs straight into `rescale`.

Both run only **once per row** (per token or per sentence), so they do not need to be fast.

## 13.1. Normalization: reduce the input to a small range

$v$ can be anything from 1 to about $2^{47}$, far too wide for a table. As in floating point, split
$v$ into a mantissa and an exponent.

**For rsqrt**, choose an **even** exponent $E$ so that $E/2$ is an integer:

$$
v = m \cdot 2^{E}, \quad m \in [1, 4)
\qquad\Rightarrow\qquad
\frac{1}{\sqrt{v}} = \frac{1}{\sqrt{m}} \cdot 2^{-E/2}
$$

**For the reciprocal:**

$$
v = m \cdot 2^{p}, \quad m \in [1, 2)
\qquad\Rightarrow\qquad
\frac{1}{v} = \frac{1}{m} \cdot 2^{-p}
$$

The $2^{-E/2}$ or $2^{-p}$ part is just a shift. What remains is $1/\sqrt{m}$ or $1/m$ with $m$ in a
small range.

The exponent comes from a **priority encoder** (`leading_one`): the position of the highest set
bit. For example, if $v$ has 40 bits then $E = 38$ (the even number $\le 39$). $m$ is obtained with
one shift and stored in Q.16.

## 13.2. Table lookup

The bits of $m$ after the leading one form the table address. Each entry holds the value at the
**center** of its interval, so the error is balanced on both sides.

| | rsqrt | reciprocal |
|---|---|---|
| Range of $m$ | $[1, 4)$ | $[1, 2)$ |
| Step | $1/256$ | $1/256$ |
| Entries | 768 | 256 |
| Value | $2^{16} / \sqrt{m}$ | $2^{16} / m$ |
| Value range | 32 776 – 65 472 | 32 800 – 65 408 |

## 13.3. One Newton step

The table gives a relative error of about $10^{-3}$. One Newton step brings it below $10^{-4}$
using only multiplies and shifts.

**rsqrt** — Newton for $f(y) = y^{-2} - m$:

$$
y \leftarrow \frac{y \cdot (3 - m y^2)}{2}
$$

In integers (Q.16):

$$
u = (y^2) \gg 16, \qquad t = (m \cdot u) \gg 16, \qquad y \leftarrow \big(y \cdot (3 \cdot 2^{16} - t)\big) \gg 17
$$

**Reciprocal** — Newton for $f(y) = 1/y - m$:

$$
y \leftarrow y \cdot (2 - m y)
$$

$$
t = (m \cdot y) \gg 16, \qquad y \leftarrow \big(y \cdot (2 \cdot 2^{16} - t)\big) \gg 16
$$

Newton converges quadratically: each step doubles the number of correct bits, from about 10 (the
table) to about 17, just enough for the 17-bit mantissa.

## 13.4. Result

$$
\frac{1}{\sqrt{v}} \approx y \cdot 2^{-(16 + E/2)}, \qquad \frac{1}{v} \approx y \cdot 2^{-(16 + p)}
$$

If the input has fraction bits ($\text{frac} = f$), the function subtracts them in the exponent:
for example $1/(\text{raw} \cdot 2^{-f}) = 2^{f} / \text{raw}$.

**rsqrt example with $v = 1000$:**

| Step | Value |
|---|---|
| $E$ | 8 (since $1000 = 3.906 \cdot 2^8$) |
| $m$ | 3.906 (Q.16: 256 000) |
| Table address | 744 |
| $y_0$ (table) | 33 151 |
| $y_1$ (after Newton) | 33 159 |
| $y_1 \cdot 2^{-20}$ | 0.0316229 |
| exact $1/\sqrt{1000}$ | 0.0316228 |

**Example in LayerNorm** ([[10 - LayerNorm#Step 5 - inverse square root of S]]):
$S = 718\,860\,868\,128$ gives $1/\sqrt{S} \approx 40525 \cdot 2^{-35}$.

## 13.5. Accuracy

| Function | Largest relative error |
|---|---|
| rsqrt | $3.6 \times 10^{-5}$ |
| reciprocal | $2.9 \times 10^{-5}$ |

Tested on 100 000 random values up to $2^{47}$ (rsqrt) and $2^{40}$ (reciprocal).

---

← [[12 - Mean pooling and cosine]] | Next: [[14 - Bit widths and FPGA mapping]] →
