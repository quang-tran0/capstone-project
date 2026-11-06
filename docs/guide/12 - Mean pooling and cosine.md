---
tags: [minilm, fpga, pooling, cosine, rag]
---

# Chapter 12 — Mean pooling and cosine

← [[11 - GELU and residual]] | Next: [[13 - rsqrt and reciprocal]] →

Code: `model/numpy/layers/mean_pooling.py`, `model/numpy/utils/similarity.py`

After the 12 encoder layers, every token has a 384-dimensional vector. This chapter combines them
into **one vector per sentence**, then compares two sentences with cosine to get the RAG score.

## 12.1. Mean pooling

### Original formula

$$
\bar{h}_d = \frac{1}{c} \sum_{i:\,M_i = 1} h_{id}, \qquad c = \sum_i M_i
$$

The average of the real tokens ($M_i = 1$), ignoring padding.

### At run time

1. **Integer sum** of the tokens with mask = 1: $\Sigma_d = \sum_i M_i [h_{id}]$ (Q.16).
2. **Multiply by the reciprocal of the token count.** $1/c$ is a `Scale` from `reciprocal_fixed`
   ([[13 - rsqrt and reciprocal]]), no division needed:

```python
inverse_count = reciprocal_fixed(counts, self.reciprocal_lut)
return inverse_count.rescale(summed, 16)          # Q.16
```

$$
[\bar{h}_d] = \operatorname{round\_shift}(\Sigma_d \cdot y,\ s), \qquad \frac{1}{c} \approx y \cdot 2^{-s}
$$

The result is the **sentence embedding** in Q.16, stored in a 32-bit register, shape $(B, 384)$.
This is what the FPGA sends back to software.

## 12.2. Cosine (RAG score)

### Original formula

$$
\operatorname{score}(a, b) = \frac{a \cdot b}{\lVert a \rVert\, \lVert b \rVert} \in [-1, 1]
$$

The closer the score is to 1, the closer the meanings. A negative score just means unrelated.

### Step 1 — Shrink each vector to 16 bits

As in step 3 of [[10 - LayerNorm]]: shift the whole vector so its largest element just fits in 16
signed bits:

$$
\hat{a} = a \gg k_a, \qquad k_a = \max\!\big(\operatorname{leading\_one}(\max |a|) + 1 - 16,\ 0\big)
$$

The scale of each vector **cancels** in the cosine (numerator and denominator are divided by the
same factor), so the shift does not change the result.

### Step 2 — Dot product and norms

$$
\mathrm{dot} = \sum_d \hat{a}_d \hat{b}_d, \qquad
\frac{1}{\lVert \hat{a} \rVert} \approx y_a \cdot 2^{-s_a}, \qquad
\frac{1}{\lVert \hat{b} \rVert} \approx y_b \cdot 2^{-s_b}
$$

The two inverse norms come from `rsqrt_fixed` applied to $\sum \hat a^2$ and $\sum \hat b^2$.

### Step 3 — Divide one at a time

Divide by $\lVert \hat{a} \rVert$ first so the intermediate stays within 64 bits, then by
$\lVert \hat{b} \rVert$:

```python
partial = inverse_a.rescale(dot, frac=15)
score = inverse_b.rescale(partial, frac=15)       # Q.15, 17-bit register
```

The result is in **Q.15**: the score lies in $[-1, 1]$, so it needs 17 signed bits.

### Note for the RAG system

Document vectors can be **normalized in advance** when building the index (divided by their
length once). At query time only the dot product and one rsqrt for the query remain.

## 12.3. Example results

Query "Hợp đồng bị vô hiệu khi nào?" ("When is a contract void?"), a few passages:

| Passage | PyTorch | NumPy | Error |
|---|---|---|---|
| Statute of limitations for contract disputes … | 0.55283 | 0.55554 | 0.0027 |
| Civil transactions by minors … declared void … | 0.48071 | 0.46857 | 0.0121 |
| Under the Civil Code, a contract is an agreement … | 0.46739 | 0.46552 | 0.0019 |
| Python is a popular programming language … | −0.21374 | −0.21906 | 0.0053 |

A negative score such as −0.21 is normal for an unrelated passage.

---

← [[11 - GELU and residual]] | Next: [[13 - rsqrt and reciprocal]] →
