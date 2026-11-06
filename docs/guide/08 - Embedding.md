---
tags: [minilm, fpga, embedding]
---

# Chapter 8 — Embedding

← [[07 - Linear]] | Next: [[09 - Self-attention and softmax]] →

Code: `model/numpy/layers/bert_embedding.py`

## 8.1. Original formula

Each token is represented by the sum of three 384-dimensional vectors, followed by LayerNorm:

$$
x = \operatorname{LayerNorm}\big(E_{\text{word}}[\text{id}] + E_{\text{pos}}[\text{position}] + E_{\text{type}}[\text{type}]\big)
$$

| Table | Size | Meaning |
|---|---|---|
| Word | $250\,037 \times 384$ | One vector per vocabulary token |
| Position | $512 \times 384$ | One vector per position in the sentence |
| Token type | $2 \times 384$ | Sentence A / sentence B (the project always uses type 0) |

## 8.2. Storage (offline)

All three tables are quantized to INT8 with **one scale per row** (see
[[06 - Quantization#6.2. Weights (offline)]]). Each row is one token's vector, so per-row scales
keep precision even for tokens with small values.

## 8.3. At run time

Read one row of INT8 codes together with that row's scale, and convert to Q.16:

$$
[e_k] = \operatorname{round\_shift}\!\left(q_k \cdot m,\ n - 16\right)
$$

```python
def _lookup(self, name, ids):
    table = self.tables[name]
    return table.scale[ids].rescale(table.codes[ids], FRAC_BITS).resize(32)
```

The multiply is 8 bits × 7 bits. Then the three vectors are added (all Q.16, so a plain addition)
and passed to LayerNorm:

$$
[x] = [e_{\text{word}}] + [e_{\text{pos}}] + [e_{\text{type}}]
$$

The position ID is just a counter $0, 1, \dots, L - 1$.

## 8.4. In hardware

The word table takes about **96 MB** (INT8), far more than the KV260's on-chip memory (about
2.9 MB). Since each token needs exactly one row, the sensible approach is to let software (ARM)
look up the table in DDR and send the vectors down, or to fetch the row with DMA. The two small
tables (position, token type) can live in BRAM.

---

← [[07 - Linear]] | Next: [[09 - Self-attention and softmax]] →
