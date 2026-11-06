---
tags: [minilm, fpga, verification]
---

# Chapter 15 — Verification and accuracy

← [[14 - Bit widths and FPGA mapping]] | Next: [[16 - Glossary]] →

## 15.1. Three levels of verification

| Level | What is checked | Tool |
|---|---|---|
| Float reference model | Does the hand-written PyTorch model match the original ONNX? | Compare with the ONNX graph itself |
| Integer model | How far is NumPy from PyTorch float32? | `model/main.py` → `report.md` |
| Individual blocks | Each integer function follows its formula; each data type follows its rules | `test_numpy_model.py` |

## 15.2. Float reference model

`model/pytorch/main.py` rebuilds MiniLM in PyTorch and loads weights from `model.onnx`. To make sure
it is correct, its output is compared with the ONNX graph itself (run with
`onnx.reference.ReferenceEvaluator`):

| Comparison | Largest absolute error |
|---|---|
| `last_hidden_state` | $1.4 \times 10^{-6}$ |
| Sentence embedding | $3.7 \times 10^{-7}$ |
| One sentence alone vs inside a padded batch | $3.6 \times 10^{-7}$ |

These are float32 rounding errors. The structure also matches: GELU uses `erf`, LayerNorm with
$\varepsilon = 10^{-12}$ after the residual, position IDs start at 0.

## 15.3. Integer versus float: `report.md`

```bash
python model/main.py
```

The command asks for a query and passages (Enter for the defaults: 20 passages, 5 of them about
civil law), then writes `model/report.md` containing:

1. **Summary table:** whether the rankings match, score error, smallest embedding cosine.
2. **Query results** of each model: passages ranked by cosine.
3. **RAG score error:** both scores, error, ranks; differing ranks are in bold.
4. **Sentence embedding error** for each sentence.
5. **Per-layer error:** a summary over the 12 layers, then a detailed table for every step inside
   each layer (Q, K, V, attention, dense, LayerNorm, GELU…).

### Error formula

$$
e_d = 100 \cdot \frac{\lvert \hat{y}_d - y_d \rvert}{\max(y) - \min(y)}\ (\%)
$$

$\hat y$ is the NumPy result (converted to float), $y$ is PyTorch. Dividing by the **range** of the
whole tensor rather than by each $y_d$ avoids huge percentages when $y_d$ is near 0 although the
actual error is small (it once produced 233% for a score of 0.004 off by 0.009).

## 15.4. Results

Query "Hợp đồng bị vô hiệu khi nào?" ("When is a contract void?"), 20 default passages:

| Quantity | INT8 with one scale per tensor (old design) | Current |
|---|---|---|
| Sentence embedding cosine | 0.847 – 0.914 | **0.9985 – 0.9997** |
| Embedding mean / max error | 7.1% / 25.6% | **0.33–0.78% / 1.3–3.7%** |
| Largest RAG score error | 0.087 | **0.012** |

**Ranking:** the top 3 are identical to PyTorch (all civil-law passages). Two pairs of passages
swap places; both pairs have very close PyTorch scores (0.008 and 0.0009 apart).

**Error across layers:** it grows from layer 0 to layer 11 and is largest in the FFN of the last
layers (max error around 9% in places). After each LayerNorm it drops back to about 1–1.7%.

| Layer (output) | Mean error | Max error | Cosine |
|---|---|---|---|
| embeddings | 0.065% | 0.36% | 0.99984 |
| layer 0 | 0.060% | 0.35% | 0.99985 |
| layer 5 | 0.18% | 1.05% | 0.99937 |
| layer 11 | 0.28% | 3.51% | 0.99871 |
| pooling | 0.49% | 2.10% | 0.99935 |

## 15.5. Unit tests

```bash
python -m unittest model.test_numpy_model -v
```

24 tests:

| Group | Checks |
|---|---|
| `FixedTest` | Width rules, signed/unsigned mixing, three overflow modes, `frac` tracking, rounding shifts, LUTs, multiplexer |
| `ScaleTest` | Configurable scale formats, quantization with a custom format, `from_peak` against float, product and rescale, `pack` |
| `IntLayerNormTest` | Against float LayerNorm, input scale cancels, constant row gives $\beta$ |
| `ArithmeticTest` | rsqrt, reciprocal, quantization, softmax, GELU, Linear, pooling, cosine against float |
| `EndToEndTest` | Whole model against PyTorch (needs `model.onnx`) |

## 15.6. Bit-exact refactoring

When the model moved from `int64` arrays to the `Fixed`/`Scale` types, every output and
intermediate trace (273 tensors, including a padded batch) was saved before and compared after:
**identical bit for bit**. Use the same method for any future change to the model that is not
meant to change its results.

---

← [[14 - Bit widths and FPGA mapping]] | Next: [[16 - Glossary]] →
