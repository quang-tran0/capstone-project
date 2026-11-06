---
tags: [minilm, fpga, fixed-point]
---

# Chapter 2 — Fixed-point numbers and Q format

← [[01 - Overview]] | Next: [[03 - Hardware data type Fixed]] →

This chapter explains how real numbers are represented with integers, the basis of every later
chapter.

## 2.1. The Q format

An integer $X$ in **Q.$f$ format** represents the real value

$$
x = X \cdot 2^{-f}
$$

$f$ is the number of **fraction bits**. Think of the binary point as fixed after the last $f$
bits, hence the name **fixed-point**.

| Real value | Q.16 (stored integer) | How |
|---|---|---|
| 1.0 | 65 536 | $1 \cdot 2^{16}$ |
| 1.5 | 98 304 | $1.5 \cdot 2^{16}$ |
| −0.25 | −16 384 | $-0.25 \cdot 2^{16}$ |
| 0.0000153 | 1 | smallest step $2^{-16}$ |

The project uses **Q.16 in a signed 32-bit register** for most tensors. The representable range is
$[-32768, 32768)$ with a step of $2^{-16} \approx 1.5 \times 10^{-5}$.

## 2.2. Addition and multiplication in Q format

**Addition:** two numbers in the same Q.$f$ add as plain integers, and the result is still Q.$f$:

$$
X_1 2^{-f} + X_2 2^{-f} = (X_1 + X_2) 2^{-f}
$$

If the $f$ differ, shift one of them first, just as you line up the decimal points when adding by hand.

**Multiplication:** the fraction bits add up:

$$
(X_1 2^{-f_1}) \cdot (X_2 2^{-f_2}) = (X_1 X_2) \cdot 2^{-(f_1 + f_2)}
$$

Example: Q.16 × Q.15 gives Q.31. To get back to Q.16, shift right by 15 bits.

## 2.3. Bit shifts

Shifting right by $k$ bits divides by $2^k$; shifting left multiplies by $2^k$. On an FPGA a shift
is almost free (just wiring, or a multiplexer).

A shift can mean two different things:

| Meaning | Effect | Example |
|---|---|---|
| **Change format** | Real value unchanged, only precision lost | Q.31 shifted right by 15 → Q.16 |
| **Divide by $2^k$** | Format unchanged, value gets smaller | Shrink a vector to fit 18 bits |

## 2.4. Rounding when shifting

A plain right shift (`>>`) **rounds down** (towards $-\infty$): $7 \gg 1 = 3$, $-7 \gg 1 = -4$.
The error can approach 1 LSB and is always biased in one direction.

The project uses **round half up**: add half a step before shifting:

$$
\operatorname{round\_shift}(x, k) =
\begin{cases}
\left(x + 2^{k-1}\right) \gg k & k > 0 \\
x & k = 0 \\
x \ll (-k) & k < 0
\end{cases}
$$

The maximum error drops to ½ LSB. In hardware this only adds a constant adder.

| $x$ | $x \gg 1$ (round down) | $\operatorname{round\_shift}(x, 1)$ |
|---|---|---|
| 3 | 1 | 2 |
| −3 | −2 | −1 |
| 5 | 2 | 3 |

There is also **round up** (ceiling): $\lceil x / 2^k \rceil = (x + 2^k - 1) \gg k$, used when
choosing scales (see [[06 - Quantization]]).

## 2.5. Bit widths

Every operation grows the number of bits:

| Operation | Needs |
|---|---|
| Add two $b$-bit numbers | 1 extra bit |
| Multiply $a$ bits by $b$ bits | $a + b$ bits |
| Accumulate $K$ numbers | $\lceil \log_2 K \rceil$ extra bits |

Example: an INT8 × INT8 product needs 16 bits; accumulating 384 of them needs $16 + 9 = 25$ bits.

On an FPGA every register has a fixed width. A value that exceeds it **overflows**: the high bits
are lost and the result is completely wrong. So the width of every signal must be known up front.
This is why the `Fixed` class of the next chapter exists.

## 2.6. Signed and unsigned

- **Signed** (two's complement), $b$ bits: range $[-2^{b-1}, 2^{b-1} - 1]$. Example INT8: $[-128, 127]$.
- **Unsigned**, $b$ bits: range $[0, 2^b - 1]$. Example UINT5: $[0, 31]$.

The project's INT8 codes use the **symmetric** range $[-127, 127]$, dropping $-128$, so the
absolute value always fits in 7 bits.

---

← [[01 - Overview]] | Next: [[03 - Hardware data type Fixed]] →
