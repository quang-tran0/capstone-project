---
tags: [minilm, fpga, index]
---

# Integer-only MiniLM on FPGA

These notes explain how the project computes the **sentence embedding** and the **RAG score**
of MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) **using integers only**, exactly the way the
hardware on the KV260 board will. The NumPy model in the repo (`model/numpy/`) is the golden
model for the RTL.

Read the chapters in order; each one builds on the previous ones.

## Part I — Foundations

1. [[01 - Overview]] — the problem, the MiniLM pipeline, why integers
2. [[02 - Fixed-point numbers and Q format]] — real numbers as integers, shifts, rounding
3. [[03 - Hardware data type Fixed]] — the `Fixed` class that models FPGA registers
4. [[04 - Scale m·2^-n]] — power-of-two scale factors, the `Scale` class

## Part II — Data

5. [[05 - Number storage]] — wide Q.16 tensors and narrow INT8 tensors, weights, bias
6. [[06 - Quantization]] — choosing scales for weights and activations, which axis a scale covers

## Part III — Model layers

7. [[07 - Linear]] — INT8 matrix multiply and conversion back to Q.16
8. [[08 - Embedding]] — INT8 table lookup and the sum of three vectors
9. [[09 - Self-attention and softmax]] — attention scores, integer softmax, context
10. [[10 - LayerNorm]] — normalization without division or square root
11. [[11 - GELU and residual]] — activation by table lookup, residual addition
12. [[12 - Mean pooling and cosine]] — one vector per sentence, RAG matching
13. [[13 - rsqrt and reciprocal]] — table lookup plus one Newton step

## Part IV — Towards hardware

14. [[14 - Bit widths and FPGA mapping]] — registers, memory, what to fix before writing RTL
15. [[15 - Verification and accuracy]] — comparison with PyTorch, reports, tests

## Appendix

- [[16 - Glossary]] — quick definitions of the terms used in these notes

> [!info] Matching code
> Repo: `capstone-project/model/`. Data types in `model/numpy/hwtypes/`, layers in
> `model/numpy/layers/`, arithmetic helpers in `model/numpy/utils/`. The code-oriented summary is
> `model/numpy/README.md`. Hardware module specs are in [[00 - Hardware Index]].
