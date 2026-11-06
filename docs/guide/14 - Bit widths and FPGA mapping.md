---
tags: [minilm, fpga, rtl, kv260]
---

# Chapter 14 — Bit widths and FPGA mapping

← [[13 - rsqrt and reciprocal]] | Next: [[15 - Verification and accuracy]] →

This chapter collects every register, lookup table and memory requirement, and assesses mapping
the datapath onto the KV260 FPGA (part `xck26-sfvc784-2LV-c`).

> [!warning] Paper assessment
> This is based on the NumPy model. There is no RTL yet and Vivado has not been run, so there are
> no timing or resource numbers. Each RTL module must pass post-place-and-route timing and
> utilization checks before it counts as done. Per-module status is tracked in [[00 - Hardware Index]].

## 14.1. Overall conclusion

The datapath **can be implemented on the FPGA**: it only uses multiplies, adds, shifts, comparisons
and table lookups, exactly what FPGAs do well. Four points need attention before writing RTL
(section 14.5).

## 14.2. Declared registers

Each row below is a `resize` call in the model ([[03 - Hardware data type Fixed#3.5. Declaring registers with resize]]).
If the data does not fit, the model raises immediately.

| Register | Width | Chapter |
|---|---|---|
| Embedding lookup output | 32 bits, Q.16 | [[08 - Embedding]] |
| INT8 codes after quantization | 8 bits | [[06 - Quantization]] |
| `Linear` accumulator | 32 bits | [[07 - Linear]] |
| `Linear` output | 32 bits, Q.16 | [[07 - Linear]] |
| Attention score accumulator | 20 bits | [[09 - Self-attention and softmax]] |
| Softmax probability codes | 8 bits | [[09 - Self-attention and softmax]] |
| Context accumulator | 22 bits | [[09 - Self-attention and softmax]] |
| Context output | 32 bits, Q.16 | [[09 - Self-attention and softmax]] |
| LayerNorm: shrunk $e$ | 18 bits | [[10 - LayerNorm]] |
| LayerNorm: $\sum e^2$ | 48 bits | [[10 - LayerNorm]] |
| LayerNorm: output | 32 bits, Q.16 | [[10 - LayerNorm]] |
| GELU: output | 32 bits, Q.16 | [[11 - GELU and residual]] |
| rsqrt / reciprocal mantissa | 17 bits unsigned | [[13 - rsqrt and reciprocal]] |
| Pooling: output | 32 bits, Q.16 | [[12 - Mean pooling and cosine]] |
| Cosine: shrunk vector / score | 16 bits / 17 bits (Q.15) | [[12 - Mean pooling and cosine]] |

## 14.3. Constants and lookup tables

| Name | Size | Format |
|---|---|---|
| $R = \lceil 2^{24}/127 \rceil$ | 1 constant: 132 105 | division by 127 when choosing scales |
| $1/\sqrt{32}$ | `Scale.const(5793, 15)` | attention scores |
| $\log_2 e$ | 1 constant: 47 274 | Q.15 |
| $1/m$ ROM (quantization) | 128 entries | 23 bits unsigned |
| EXP2 (softmax) | 257 entries | 16 bits, Q.15 |
| PHI (GELU) | 1024 entries | 16 bits, Q.15 |
| RSQRT | 768 entries | 17 bits, Q.16 |
| RECIP (reciprocal) | 256 entries | 17 bits, Q.16 |

All of them are small and fit in LUTRAM or one BRAM each.

## 14.4. What maps well onto DSP48E2

A DSP48E2 multiplies $27 \times 18$ bits and accumulates 48 bits.

| Operation | Width | Note |
|---|---|---|
| INT8 × INT8 MAC | acc $\le 25$ bits | Two INT8 MACs can be packed into one DSP |
| Rescale `acc · m_x m_w` | $25 \times 14$ bits → 39 bits | Fits 1 DSP |
| `m_x · m_w` | $7 \times 7$ bits | LUTs or a small DSP |
| LayerNorm $e^2$, accumulate | $18 \times 18$, 48 bits | Exactly the DSP size |
| LayerNorm `e·m`, `norm·G` | $18 \times 17$, $18 \times 18$ | 1 DSP each |
| Attention scores, context | 20 / 22 bits | Small |

## 14.5. What to fix before writing RTL

### (1) Wide multiply in quantization

Every input element of every `Linear` computes $x \cdot \mathrm{RECIP}[m]$: a 32-bit $x$ times
23 bits gives **about 55 bits**, needing 3–4 DSPs per element. It runs for every element (384 or
1536 per token), so it is expensive.

**Fix:** the output code has only 8 bits, so $x$ can be shifted down before the multiply (knowing
$n$ tells how many low bits can be dropped), making both factors fit a DSP. The **NumPy model must
be changed first** to stay bit-exact with the RTL.

Similarly, $P \cdot R$ when choosing the scale needs about 57 bits, but only once per token, so it
can be done sequentially.

### (2) Operations that scan a row several times

- Per-token quantization needs the **peak of the whole row** before the first code can be computed.
- Softmax needs the row max and sum.
- LayerNorm needs the sum, the peak and the sum of squares.

Each needs the whole row buffered: $384 \times 32$ bits = 1.5 KB, or $1536 \times 32$ bits = 6 KB
for the dense2 input. BRAM holds this easily; the cost is extra latency or a two-phase pipeline.

### (3) Memory: weights do not fit on chip

| Data | Size (INT8) |
|---|---|
| Word embedding table | ≈ 96 MB |
| Encoder weights, per layer | ≈ 1.77 MB |
| Encoder weights, 12 layers | ≈ 21 MB |
| XCK26 on-chip memory (144 BRAM36 + 64 URAM) | ≈ 2.9 MB |

Weights must be **streamed from DDR4** over AXI, layer by layer or tile by tile. One layer
(1.77 MB) fits in URAM but leaves no room for double buffering. The embedding lookup is best done
in software.

### (4) Per-element shifter

The shift amount $n_x + n_w - 16$ changes per channel, so every output lane needs a **barrel
shifter**. With hundreds of parallel lanes this costs a noticeable number of LUTs. Since $n_w$ lies
in a narrow range, normalizing all channels offline to a common $n_w$ would make the shift
constant across a row and almost remove the barrel shifters. This also has to be changed in NumPy first.

## 14.6. Rough speed estimate

A 64-token sentence through 12 layers needs about **2.7 billion MACs**. A 1024-MAC systolic array
(about 512 DSPs with two MACs per DSP) at 200 MHz takes **about 13 ms** of compute. Reading 21 MB
of weights from DDR adds a few ms. The KV260 has 1248 DSPs, so this is feasible, but it is only an estimate.

## 14.7. Suggested order of work

1. Change the NumPy model: narrow the two wide multiplies, and normalize $n_w$ if the barrel
   shifters are to be removed.
2. Rerun the PyTorch comparison ([[15 - Verification and accuracy]]) to make sure accuracy does not drop.
3. Write the RTL block by block, using the NumPy model as the testbench reference (bit-exact).
4. Run timing (setup, hold, pulse width) and utilization after place-and-route on
   `xck26-sfvc784-2LV-c` at the target clock for every RTL block.

---

← [[13 - rsqrt and reciprocal]] | Next: [[15 - Verification and accuracy]] →
