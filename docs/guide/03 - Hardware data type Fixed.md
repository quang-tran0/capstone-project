---
tags: [minilm, fpga, hwtypes]
---

# Chapter 3 — Hardware data type `Fixed`

← [[02 - Fixed-point numbers and Q format]] | Next: [[04 - Scale m·2^-n]] →

Code: `model/numpy/hwtypes/fixed.py`

## 3.1. Why a dedicated data type

If the model is written with NumPy `int64` arrays, every value has 64 bits and nobody knows how
many bits a signal really needs. A multiplication can produce 55 bits unnoticed, and the 32-bit
register overflow is only discovered when writing RTL.

`Fixed` solves this: each array carries its **register format**, and every operation computes the
width of its result following HDL rules.

## 3.2. Structure

```python
Fixed(raw, bits, frac=0, signed=True, overflow="error", name=None)
```

| Attribute | Meaning |
| --- | --- |
| `raw` | The integer values (stored as `np.int64` so NumPy does not overflow while computing) |
| `bits` | Register width including the sign bit (1 to 63) |
| `signed` | Signed (two's complement) or unsigned |
| `frac` | Fraction bits: real value $= \text{raw} \cdot 2^{-\text{frac}}$ |

The format is only metadata. Every operation works on `raw`, exactly as RTL works on bits.

**Creating values:**

| Constructor | Use when |
| --- | --- |
| `Fixed(raw, bits, frac, signed)` | You already have the integer values |
| `Fixed.from_float(x, bits, frac)` | Offline: round real values to the nearest code (bias, gamma, beta) |
| `Fixed.const(v)` | Constants; picks the smallest width that fits |
| `x.to_float()` | Debugging and traces only |

**Predefined types:**

| Type | Format |
|---|---|
| `Int8`, `Int16`, `Int32` | signed, 8 / 16 / 32 bits |
| `UInt8` | unsigned 8 bits |
| `Q16` | signed 32 bits, 16 fraction bits |

New types are made with `fixed_type`:

```python
from model.numpy.hwtypes import Int8, Q16, fixed_type

a = Int8([100, -50])
x = Q16.from_float([1.5, -0.25])        # raw = [98304, -16384]
UInt5 = fixed_type("UInt5", 5, signed=False)
```

## 3.3. Operations and result widths

Operators are overloaded. Results **always have enough bits** and are never truncated implicitly:

| Operation | Result width | Result `frac` |
|---|---|---|
| `a + b`, `a - b` | one bit wider than the wider operand | both sides must match, otherwise an error |
| `-a` | one extra bit | unchanged |
| `abs(a)` | unchanged, result unsigned | unchanged |
| `a * b` | `a.bits + b.bits` | `a.frac + b.frac` |
| `a @ b` (MAC over $K$ elements) | `a.bits + b.bits` $+ \lceil \log_2 K \rceil$ | added |
| `a.sum(axis)` over $n$ elements | `bits` $+ \lceil \log_2 n \rceil$ | unchanged |
| `a.max()`, `a.min()` | unchanged | unchanged |

```python
a, b = Int8([-128, 127]), Int8([127, -128])
(a + b).bits     # 9
(a * b).bits     # 16
acc = Int8(...) @ Int8(...)   # K = 384 -> 8 + 8 + 9 = 25 bits
```

**Adding two numbers with different `frac` raises an error.** This is deliberate: in hardware the
designer must align the binary points explicitly, so the model forces the same.

**Mixing signed and unsigned:** the result is signed if either operand is. A $u$-bit unsigned
number is treated as a $(u + 1)$-bit signed number.

**Python constants** behave like HDL literals: in addition they take the other operand's `frac`;
in multiplication they are plain integers.

## 3.4. Shifts

| Operation | Meaning | Result `frac` |
|---|---|---|
| `a >> k`, `a << k` | Raw bit shift (`>>` rounds down) | unchanged |
| `a.round_shift(k)` | Shift right, round half up; $k < 0$ shifts left | `frac - k` |
| `a.floor_shift(k)` | Shift right, round down; $k < 0$ shifts left | `frac - k` |
| `a & mask` | Keep the low bits | unchanged |

The two groups match the two meanings of a shift in [[02 - Fixed-point numbers and Q format#2.3. Bit shifts]]:

- `>>`, `<<` **divide/multiply by $2^k$**: `frac` kept, value changes.
- `round_shift`, `floor_shift` **change format**: value kept, `frac` drops by $k$.

```python
x = Q16.from_float([1.5])                    # frac 16
y = Fixed.from_float([0.5], 16, frac=13)     # frac 13
p = x * y                                    # 48 bits, frac 29
p.round_shift(13).to_float()                 # [0.75], frac 16
```

When the shift amount is an array (each element shifted differently), there is no single `frac`
to infer, so the result `frac=` must be passed explicitly.

## 3.5. Declaring registers with resize

```python
acc = (x.codes @ w.codes).resize(32, name="accumulator")
```

`resize` is where a **register of fixed width is declared**. If the data does not fit:

| `overflow` | Behavior |
|---|---|
| `"error"` (default) | Raise `OverflowError` with the register name and the number of bits actually needed |
| `"wrap"` | Drop the high bits (two's-complement wrap, like real hardware) |
| `"saturate"` | Clamp to the largest / smallest value |

```python
Int8([128])                            # OverflowError
Int8([128, -129], overflow="wrap")     # [-128, 127]
Int8([300, -300], overflow="saturate") # [127, -128]
```

The workflow: operations compute full-width results, and wherever the hardware has a register,
call `resize`. This way the model catches every overflow as soon as it runs.

Two helpers: `with_frac(f)` reinterprets `frac` without changing any bit; `required_bits()` reports
how many bits the current data actually needs.

## 3.6. Small logic blocks

| Function | Hardware equivalent |
|---|---|
| `x.clip(lo, hi)` | Saturator; the width shrinks to fit $[lo, hi]$ |
| `where(cond, a, b)` | Multiplexer |
| `leading_one(x)` | Priority encoder: bits needed to represent $x \ge 0$ |
| `lut[address]` | ROM/LUT read; an out-of-range address raises `IndexError` |
| `x.max(axis, where=mask)` | Comparator tree that skips masked elements |

## 3.7. Limitations

- The maximum width is 63 bits (the limit of `int64`). If the rules give more than 63 bits,
  `Fixed` uses the actual width of the data; if the data really needs more than 63 bits, it raises.
- For shifts by an array of amounts, the result width is derived from the range of amounts present
  in the data, not from their declared width. Registers declared with `resize` are always checked
  strictly.

---

← [[02 - Fixed-point numbers and Q format]] | Next: [[04 - Scale m·2^-n]] →
