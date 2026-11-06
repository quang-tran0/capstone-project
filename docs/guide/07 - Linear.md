---
tags: [minilm, fpga, linear]
---

# Chapter 7 — Linear (matrix multiply)

← [[06 - Quantization]] | Next: [[08 - Embedding]] →

Code: `model/numpy/layers/linear.py`

`Linear` accounts for almost all of MiniLM's computation: each encoder layer has 6 of them (Q, K,
V, attention output, FFN dense1, FFN dense2). On the FPGA this is the job of the **systolic array**.

## 7.1. From the original formula to INT8 MACs

Original formula, with $K = 384$ or $1536$:

$$
y_{ij} = \sum_{k=1}^{K} x_{ik} W_{kj} + b_j
$$

Substitute $x_{ik} \approx x_{q,ik} \cdot S_{x,i}$ (per-token scale) and
$W_{kj} \approx W_{q,kj} \cdot S_{w,j}$ (per-column scale). Neither scale depends on $k$, so both
move outside the sum:

$$
y_{ij} \approx S_{x,i}\, S_{w,j} \underbrace{\sum_{k} x_{q,ik}\, W_{q,kj}}_{\mathrm{acc}_{ij}} + b_j
$$

So the systolic array **only does INT8 multiply-accumulate**:

$$
\mathrm{acc}_{ij} = \sum_{k=1}^{K} x_{q,ik}\, W_{q,kj}
$$

The rest (scale multiply, bias add) happens at the array output, once per result element.

## 7.2. Converting the result to Q.16

With $S = m \cdot 2^{-n}$:

$$
S_{x,i} S_{w,j} = (m_{x,i}\, m_{w,j}) \cdot 2^{-(n_{x,i} + n_{w,j})}
$$

For a Q.16 result (an extra factor $2^{16}$):

$$
[y_{ij}] = \operatorname{round\_shift}\!\Big(\mathrm{acc}_{ij} \cdot (m_{x,i}\, m_{w,j}),\ n_{x,i} + n_{w,j} - 16\Big) + [b_j]
$$

In code it is a single line:

```python
output = (x.scale * self.weight.scale).rescale(accumulator, FRAC_BITS) + self.bias
```

Why multiplying by $m$ and shifting by $n$ is enough is explained in
[[04 - Scale m·2^-n#Rescale - multiply by the scale and change format]].

## 7.3. The whole procedure

```python
def __call__(self, x):                                   # x: Q.16 (B, L, K)
    x = quantize(x, axis=-1)                             # INT8 codes + per-token scale
    acc = (x.codes @ self.weight.codes).resize(32)       # systolic array
    y = (x.scale * self.weight.scale).rescale(acc, 16)   # × m, shift n
    return (y + self.bias).resize(32)                    # Q.16
```

1. **Quantize the input** per token ([[06 - Quantization#6.3. Activations (at run time)]]).
2. **INT8 MAC** on the systolic array into a 32-bit register.
3. **Rescale:** multiply by $m_x m_w$ (14 bits), shift right with rounding.
4. **Add the bias** in Q.16 and store in a 32-bit register.

In self-attention, $Q$, $K$ and $V$ share **one** quantization of their input.

## 7.4. A complete worked element

$X \cdot W_Q$ in layer 0, sentence "Xin chào FPGA" (7 tokens), element $Y[\text{token } 1, \text{channel } 0]$:

**How many scales does this multiply use?** $7 + 384 = 391$ pairs $(m, n)$:

| Operand | Shape | Scale | Number of scales |
|---|---|---|---|
| $X$ (input) | $(1, 7, 384)$ | one per token | 7 |
| $W_Q$ | $(384, 384)$ | one per column | 384 |
| $Y$ (output) | $(1, 7, 384)$ | none, it is Q.16 | 0 |

**Scales of $X$** (computed at run time):

```
x_m = [119  76  92  86  84  68 116]
x_n = [ 11  12  12  11  12  12  11]
```

**Scales of $W_Q$** (computed offline): channel 0 has $m = 80$, $n = 16$, i.e.
$S_w = 80 \cdot 2^{-16} = 0.0012207$.

**Compute:**

```
acc   = Σ x_q[1,k] · w_q[k,0]                 = -26375       (sum of 384 INT8 products)
m     = x_m[1] · w_m[0] = 76 · 80             = 6080         (7 bits × 7 bits = 13 bits)
shift = x_n[1] + w_n[0] − 16 = 12 + 16 − 16  = 12
y     = round_shift(-26375 · 6080, 12) + bias
      = -39150 + (-1202)                       = -40352       (Q.16)  →  -0.61572
```

Floating-point check: $-26375 \times 0.018555 \times 0.0012207 - 0.01834 = -0.6157$. It matches.

Element $Y[i, j]$ combines the scale of **row $i$** with the scale of **column $j$**. For each
result element the hardware needs: 1 multiply $m_x m_w$, 1 multiply by acc, 1 shift, 1 bias add.

## 7.5. Bit widths

$$
|\mathrm{acc}| \le K \cdot 127^2 < \begin{cases} 2^{23} & K = 384 \\ 2^{25} & K = 1536 \end{cases}
$$

| Signal | Width |
|---|---|
| $x_q$, $W_q$ | 8 bits |
| $\mathrm{acc}$ | 32-bit register (data $\le 26$ bits) |
| $m_x m_w$ | 14 bits |
| $\mathrm{acc} \cdot m_x m_w$ | $\le 40$ bits (fits one DSP48E2: 25 × 14 bits, 48-bit P) |
| bias | Q.16, 18 bits suffice |
| $[y]$ | 32-bit register, Q.16 |

> [!warning] Note for the RTL
> The product $\mathrm{acc} \cdot m$ needs about 40 bits before the shift, even though the final
> result fits in 32 bits. A multiplier with a 32-bit output would overflow.

---

← [[06 - Quantization]] | Next: [[08 - Embedding]] →
