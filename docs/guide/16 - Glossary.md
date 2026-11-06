---
tags: [minilm, fpga, glossary]
---

# Appendix — Glossary

← [[15 - Verification and accuracy]] | Back to: [[00 - Index]]

| Term | Meaning | See |
|---|---|---|
| **Accumulator** | Register that sums the results of many multiplications | [[07 - Linear]] |
| **Barrel shifter** | Shifter whose shift amount can change at run time | [[14 - Bit widths and FPGA mapping]] |
| **BRAM / URAM** | On-chip FPGA memory blocks (36 Kb / 288 Kb each) | [[14 - Bit widths and FPGA mapping]] |
| **Code** | Integer stored in place of a real value after quantization | [[06 - Quantization]] |
| **Cosine similarity** | Similarity of two vectors, in $[-1, 1]$ | [[12 - Mean pooling and cosine]] |
| **DSP48E2** | Xilinx FPGA block: $27 \times 18$-bit multiply + 48-bit accumulate | [[14 - Bit widths and FPGA mapping]] |
| **Fixed-point** | Real numbers represented by integers with a fixed binary point | [[02 - Fixed-point numbers and Q format]] |
| **Golden model** | Reference model whose results the RTL must reproduce | [[01 - Overview]] |
| **LSB** | Least significant bit; 1 LSB of Q.16 is $2^{-16}$ | [[02 - Fixed-point numbers and Q format]] |
| **LUT / ROM** | Table of precomputed values | [[11 - GELU and residual]] |
| **MAC** | Multiply-accumulate | [[07 - Linear]] |
| **Mask** | Marks real tokens (1) and padding tokens (0) | [[09 - Self-attention and softmax]] |
| **Newton's method** | Iterative refinement; each step doubles the number of correct bits | [[13 - rsqrt and reciprocal]] |
| **Offline** | Preparation on a PC, where floats are allowed | [[01 - Overview]] |
| **Outlier** | A channel with values much larger than the others | [[06 - Quantization]] |
| **Overflow** | A value exceeds the register width and the high bits are lost | [[02 - Fixed-point numbers and Q format]] |
| **Peak** | Largest absolute value of a row or column | [[06 - Quantization]] |
| **Priority encoder** | Circuit that finds the highest set bit (`leading_one`) | [[13 - rsqrt and reciprocal]] |
| **Q.$f$** | Integer representing the value $X \cdot 2^{-f}$ | [[02 - Fixed-point numbers and Q format]] |
| **Quantization** | Converting real values to integer codes plus a scale | [[06 - Quantization]] |
| **RAG** | Retrieval-Augmented Generation: retrieve relevant documents to answer | [[01 - Overview]] |
| **Register** | Storage element of fixed bit width; `resize` in the model | [[03 - Hardware data type Fixed]] |
| **Rescale** | Multiply by $m$ and shift by $n$ to bring a result to Q.16 | [[04 - Scale m·2^-n]] |
| **Residual** | Adding a block's input to its output | [[11 - GELU and residual]] |
| **round_shift** | Right shift with round-half-up | [[02 - Fixed-point numbers and Q format]] |
| **RTL** | Hardware description in Verilog/SystemVerilog | [[14 - Bit widths and FPGA mapping]] |
| **Saturate** | Clamp to the allowed range instead of overflowing | [[03 - Hardware data type Fixed]] |
| **Scale** | Factor $m \cdot 2^{-n}$ attached to integer codes | [[04 - Scale m·2^-n]] |
| **Sentence embedding** | 384-dimensional vector representing a sentence's meaning | [[12 - Mean pooling and cosine]] |
| **Systolic array** | Grid of multiply-accumulate cells for matrix multiplication | [[07 - Linear]] |
| **Token** | Piece of text produced by the tokenizer (a word or part of a word) | [[08 - Embedding]] |
| **Wide / narrow tensor** | Q.16 32-bit / INT8 + scale | [[05 - Number storage]] |

---

← [[15 - Verification and accuracy]] | Back to: [[00 - Index]]
