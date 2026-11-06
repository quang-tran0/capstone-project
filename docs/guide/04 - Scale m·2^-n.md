---
tags: [minilm, fpga, hwtypes, scale]
---

# Chapter 4 — Scale $m \cdot 2^{-n}$

← [[03 - Hardware data type Fixed]] | Next: [[05 - Number storage]] →

Code: `model/numpy/hwtypes/scale.py`

## 4.1. What a scale is

When a tensor is stored as INT8 codes, a **scale factor** $S$ says which real value each code
stands for:

$$
x \approx x_q \cdot S, \qquad x_q \in [-127, 127]
$$

Example: a row whose largest value is 2.34 gets $S \approx 2.34 / 127 = 0.0184$. Code 127 stands
for 2.34, code 20 for 0.37.

## 4.2. Why the form $m \cdot 2^{-n}$

A scale is a real number, and the FPGA does not want to deal with real numbers. The project writes
every scale as:

$$
S = m \cdot 2^{-n}, \qquad m, n \text{ unsigned integers}
$$

Multiplying by $S$ then needs only **one multiplier** (by $m$) and **one shifter** (right by $n$
bits). No floating point, no division.

Example: $S = 0.018555 \approx 76 \cdot 2^{-12}$ ($76 / 4096 = 0.018555$).

## 4.3. Default format: 7 + 5 bits

$$
m \in [1, 127]\ (7 \text{ bits}), \qquad n \in [0, 31]\ (5 \text{ bits})
$$

- **$m$ is chosen as large as possible** (for MiniLM weights it always lies in $[64, 127]$), so it
  always carries 7 significant bits.
- **$n$ needs 5 bits** because weight scales are typically $10^{-3}$–$10^{-4}$, smaller than
  $2^{-15}$. On MiniLM, $n$ for weights lies in 11..18 and for the word embedding table in 14..22.
  With only 4 bits ($n \le 15$), small scales would get an $m$ of just a few units and lose precision.
- Together a scale takes **12 bits**.

## 4.4. `ScaleFormat`: configurable widths

```python
ScaleFormat(m_bits=7, n_bits=5)
```

The widths of both fields can be changed to try different hardware options:

| Attribute / method | Meaning |
|---|---|
| `m_max`, `n_max` | $2^{m\_bits} - 1$ and $2^{n\_bits} - 1$ (127 and 31 by default) |
| `scale(m, n)` | Build a `Scale`, checking both fields fit the format |
| `from_float(s)` | Offline: choose $(m, n)$ for a real number $s$ |
| `from_peak(peak, code_max)` | Run time, integers only: choose $(m, n) \approx \text{peak} / \text{code\_max}$ |

```python
fmt = ScaleFormat(m_bits=4, n_bits=3)    # m <= 15, n <= 7
q = quantize(x, scale_format=ScaleFormat(5, 4), code_bits=6)
```

How $(m, n)$ is chosen is explained in [[06 - Quantization]].

## 4.5. `Scale`: operations

`Scale(m, n)` holds two unsigned `Fixed` fields. They have a shape, so one `Scale` can hold many
scales (one per token, one per column) and broadcasts during computation.

| Operation | Meaning |
| --- | --- |
| `S1 * S2` | $m = m_1 m_2$, $n = n_1 + n_2$; full width, no rounding |
| `S.rescale(value, frac)` | Attach scale $S$ to the integer `value`: returns $\text{value} \cdot m \cdot 2^{-n}$ in Q.`frac` (see below) |
| `S[idx]`, `S.swapaxes(...)` | Select rows, swap axes like an array |
| `S.pack()` | Concatenate $\{m, n\}$ into one word (12 bits by default) |
| `Scale.const(m, n)` | A constant, e.g. $1/\sqrt{32} \approx$ `Scale.const(5793, 15)` |

### Multiplying two scales

$$
(m_1 2^{-n_1}) \cdot (m_2 2^{-n_2}) = (m_1 m_2) \cdot 2^{-(n_1 + n_2)}
$$

In hardware: one 7 × 7-bit multiplier (14-bit result) and one 5-bit adder.

### Rescale - multiply by the scale and change format

#### What `value` is

`rescale` always involves **two different things**:

| Thing | What it is | Example |
| --- | --- | --- |
| `value` | **Data**: an integer (or array of integers) with no real meaning yet | Accumulator after a matrix multiply, INT8 codes of the embedding table |
| `self` (scale $S$) | **Factor** saying how much one unit of `value` is worth | $S = m \cdot 2^{-n}$ |

$m$ and $n$ belong to the **scale**, not to `value`. The real value represented by `value` is:

$$
\text{real value} = \text{value} \times S = \text{value} \times m \times 2^{-n}
$$

Think of an invoice: `value` is the **quantity** (say 26 375 items) and $S$ is the **unit price**.
The total is the product of the two. `rescale` computes exactly that product and returns it in
Q.16, so later steps share one unit.

Example in `Linear` ([[07 - Linear]]):

- The systolic array produces $\text{acc} = -26375$, a sum of INT8 × INT8 products with no clear unit.
- The matching scale is $S = S_x \cdot S_w = 6080 \cdot 2^{-28}$ (input scale times weight scale).
- The real value is $-26375 \times 6080 \times 2^{-28} = -0.5974$.

Here `value = acc`, while `m = 6080` and `n = 28` belong to the scale.

#### What rescale has to do

Input: the integer `value` and the scale $(m, n)$.
Output: **a new integer in Q.16** representing the same real value, i.e. an integer $y$ with

$$
y \cdot 2^{-16} \approx \text{value} \cdot m \cdot 2^{-n}
\qquad\Longleftrightarrow\qquad
y \approx \text{value} \cdot m \cdot 2^{16 - n}
$$

The right-hand side has only two operations:

1. **Multiply `value` by $m$:** integer multiplication, exact.
2. **Multiply by $2^{16 - n}$:** i.e. **shift right by $n - 16$ bits** (with rounding). If $n < 16$
   it is a left shift.

$$
y = \operatorname{round\_shift}\big(\text{value} \cdot m,\ n - 16\big)
$$

#### Why shifting by $n$ alone is enough

Because $m$ has already been **multiplied into `value`** in step 1, nothing of $m$ is left to
handle afterwards. Follow "how much one unit of the integer is worth" through the steps, with the
example above:

| Step | Integer held | One unit is worth | Real value |
|---|---|---|---|
| Start | `value` = −26 375 | $S = 6080 \cdot 2^{-28}$ | $-26375 \times 6080 \times 2^{-28} = -0.5974$ |
| After step 1: × $m$ | `value · m` = −160 360 000 | $2^{-28}$ | $-160\,360\,000 \times 2^{-28} = -0.5974$ |
| After step 2: shift right $28 - 16 = 12$ | $y$ = −39 150 | $2^{-16}$ (Q.16) | $-39150 \times 2^{-16} = -0.5974$ |

The last column never changes: each step only changes the representation, not the value. Step 1
moves $m$ from the "unit price" into the "quantity", so the unit price is just a power of two.
Converting between powers of two only needs a shift.

Step 1 is exact (integer times integer). The only error is in step 2, at most ½ LSB of Q.16
thanks to rounding.

#### The `frac` and `value.frac` parameters

The code is slightly more general than the formula above:

```python
def rescale(self, value, frac):
    return (value * self.m).round_shift(self.n + (value.frac - frac), frac=frac)
```

| Parameter | Meaning |
|---|---|
| `frac` | Desired output format Q.`frac`. Usually 16; cosine uses 15; LayerNorm uses 17 |
| `value.frac` | If `value` is itself a fixed-point number (not a plain integer), its fraction bits are included in the shift |

The general shift is $n + \text{value.frac} - \text{frac}$. For an accumulator or INT8 codes,
`value.frac = 0` and `frac = 16`, so the shift is $n - 16$ as above.

#### Example in code

```python
acc = Fixed.const(-26375)                            # value: an integer, frac = 0
scale = Scale.const(76, 12) * Scale.const(80, 16)   # S_x * S_w: m = 6080, n = 28
y = scale.rescale(acc, frac=16)
# step 1: -26375 * 6080      = -160 360 000
# step 2: round_shift(..., 28 + 0 - 16 = 12) = -39150     (Q.16)
y.to_float()                                         # -0.5974
```

#### Where rescale is used

| Place | `value` | Scale $(m, n)$ |
|---|---|---|
| `Linear` ([[07 - Linear]]) | INT8 × INT8 accumulator | $S_x \cdot S_w$: token scale × weight column scale |
| Embedding ([[08 - Embedding]]) | INT8 codes of one table row | That row's scale |
| Attention scores ([[09 - Self-attention and softmax]]) | $QK^\top$ accumulator | $S_Q \cdot S_K \cdot (1/\sqrt{32})$ |
| Context ([[09 - Self-attention and softmax]]) | $PV$ accumulator | $S_V \cdot \frac{1}{127E}$ |
| Mean pooling ([[12 - Mean pooling and cosine]]) | Sum over tokens (Q.16) | $1/c$ |
| LayerNorm ([[10 - LayerNorm]]) | Shrunk deviation $e$ | $1/\sqrt{S}$ |

Every case follows one pattern: **there is an integer that needs a unit attached, and a scale
that says what that unit is**.

## 4.6. `QTensor`

`QTensor(codes, scale)` pairs **integer codes** (usually INT8) with their **scale**. This is the
form of the data entering every matrix multiply. `dequantize()` returns real values, for
debugging only.

---

← [[03 - Hardware data type Fixed]] | Next: [[05 - Number storage]] →
