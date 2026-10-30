"""Fixed-width integer arrays that follow FPGA register semantics.

A Fixed holds raw two's-complement integers (np.int64) together with the
register format: total width `bits`, `signed`, and `frac` fraction bits
(value = raw * 2**-frac). The format is only metadata; every operation is
done on the raw integers, exactly as the RTL would.

Width rules, as in HDL with full-precision results:
- a + b, a - b: one bit wider than the wider operand.
- a * b: a.bits + b.bits; fractions add.
- a @ b, sum over n terms: product width + ceil(log2(n)).
- Shifts by a tensor of amounts use the range of amounts actually present.
Results are never truncated implicitly. `resize()` declares a register of a
given width and raises (or wraps / saturates) when the data does not fit.

Plain Python ints and integer NumPy arrays act as HDL literals: in + and -
they take the other operand's fraction; in * they are integers (frac 0).
"""

import math

import numpy as np


# Widest format whose raw values (and a rounding offset) still fit int64.
MAX_BITS = 63

OVERFLOW_MODES = ("error", "wrap", "saturate")


def format_range(bits, signed):
    """Inclusive raw range of a register."""
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def bits_for(low, high, signed):
    """Smallest register width holding [low, high]."""
    if not signed and low < 0:
        raise ValueError("Negative values need a signed format")
    if signed:
        return max(int(high).bit_length(), int(-low - 1).bit_length() if low < 0 else 0) + 1
    return max(int(high).bit_length(), 1)


def _clog2(n):
    return max(int(n) - 1, 0).bit_length()


def _shift_range(shift):
    """(min, max) of a shift amount given as int, array or Fixed."""
    raw = shift.raw if isinstance(shift, Fixed) else np.asarray(shift, dtype=np.int64)
    if raw.size == 0:
        return 0, 0
    return int(raw.min()), int(raw.max())


def _raw(value):
    return value.raw if isinstance(value, Fixed) else np.asarray(value, dtype=np.int64)


class Fixed:
    """Array of fixed-width integers with a fixed-point fraction."""

    __slots__ = ("raw", "bits", "frac", "signed")
    __array_ufunc__ = None  # numpy defers to Fixed's reflected operators

    def __init__(self, raw, bits, frac=0, signed=True, overflow="error", name=None):
        raw = np.asarray(raw)
        if raw.dtype == bool:
            raw = raw.astype(np.int64)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("Fixed needs integer raw values; use Fixed.from_float for reals")
        if not 1 <= bits <= MAX_BITS:
            raise ValueError(f"Fixed width must be in [1, {MAX_BITS}], got {bits}")
        if overflow not in OVERFLOW_MODES:
            raise ValueError(f"overflow must be one of {OVERFLOW_MODES}")
        raw = raw.astype(np.int64)
        low, high = format_range(bits, signed)
        if raw.size and (raw.min() < low or raw.max() > high):
            if overflow == "error":
                label = f"{name} " if name else ""
                kind = "signed" if signed else "unsigned"
                raise OverflowError(
                    f"{label}needs {bits_for(int(raw.min()), int(raw.max()), True)} signed bits, "
                    f"register is {bits} {kind} bits"
                )
            if overflow == "wrap":
                raw = (raw - low) % (1 << bits) + low
            else:
                raw = np.clip(raw, low, high)
        self.raw = raw
        self.bits = int(bits)
        self.frac = int(frac)
        self.signed = bool(signed)

    # ----- construction and conversion -----

    @classmethod
    def const(cls, value, frac=0):
        """Literal with the smallest width that holds its values."""
        raw = np.asarray(value, dtype=np.int64)
        low, high = (int(raw.min()), int(raw.max())) if raw.size else (0, 0)
        signed = low < 0
        return cls(raw, bits_for(low, high, signed), frac, signed)

    @classmethod
    def from_float(cls, value, bits, frac=0, signed=True, overflow="error", name=None):
        """Offline only: round real values to the nearest code (ties to even)."""
        raw = np.rint(np.asarray(value, dtype=np.float64) * 2.0**frac).astype(np.int64)
        return cls(raw, bits, frac, signed, overflow, name)

    def to_float(self):
        """Debug/trace only."""
        return self.raw.astype(np.float64) * 2.0**-self.frac

    def resize(self, bits, signed=None, overflow="error", name=None):
        """Store into a register of the given width; the fraction is kept."""
        signed = self.signed if signed is None else signed
        return Fixed(self.raw, bits, self.frac, signed, overflow, name)

    def with_frac(self, frac):
        """Reinterpret the same bits with another fraction (no data change)."""
        return Fixed(self.raw, self.bits, frac, self.signed)

    def required_bits(self):
        """Width the current data actually needs in this signedness."""
        if self.raw.size == 0:
            return 1
        return bits_for(int(self.raw.min()), int(self.raw.max()), self.signed)

    def _signed_bits(self):
        return self.bits if self.signed else self.bits + 1

    def _coerce(self, other, frac):
        if isinstance(other, Fixed):
            return other
        return Fixed.const(other, frac)

    # ----- array protocol -----

    @property
    def shape(self):
        return self.raw.shape

    @property
    def ndim(self):
        return self.raw.ndim

    @property
    def size(self):
        return self.raw.size

    def __len__(self):
        return len(self.raw)

    def __repr__(self):
        kind = "S" if self.signed else "U"
        return f"Fixed<{kind}{self.bits}.{self.frac}>(shape={self.shape})"

    def _like(self, raw):
        return Fixed(raw, self.bits, self.frac, self.signed)

    def __getitem__(self, key):
        """NumPy indexing; a Fixed index is a LUT/RAM address and must be in range."""
        if isinstance(key, Fixed):
            address = key.raw
            if address.size and (address.min() < 0 or address.max() >= len(self.raw)):
                raise IndexError("Address outside the table")
            key = address
        return self._like(self.raw[key])

    def reshape(self, *shape):
        return self._like(self.raw.reshape(*shape))

    def transpose(self, *axes):
        return self._like(self.raw.transpose(*axes))

    def swapaxes(self, axis1, axis2):
        return self._like(self.raw.swapaxes(axis1, axis2))

    def broadcast_to(self, shape):
        return self._like(np.broadcast_to(self.raw, shape))

    # ----- arithmetic -----

    def _check_sim(self, bits, tight):
        """Fall back to the data's real width if the static bound exceeds int64."""
        if bits <= MAX_BITS:
            return bits
        if tight > MAX_BITS:
            raise OverflowError(f"Result needs {tight} bits; int64 cannot simulate it")
        return tight

    def _add(self, other, subtract=False, reverse=False):
        other = self._coerce(other, self.frac)
        if other.frac != self.frac:
            raise ValueError(f"Align fractions before adding: Q.{self.frac} and Q.{other.frac}")
        signed = subtract or self.signed or other.signed
        if signed:
            bits = max(self._signed_bits(), other._signed_bits()) + 1
        else:
            bits = max(self.bits, other.bits) + 1
        bits = self._check_sim(bits, max(self.required_bits(), other.required_bits()) + 2)
        left, right = (other, self) if reverse else (self, other)
        raw = left.raw - right.raw if subtract else left.raw + right.raw
        return Fixed(raw, bits, self.frac, signed)

    def __add__(self, other):
        return self._add(other)

    __radd__ = __add__

    def __sub__(self, other):
        return self._add(other, subtract=True)

    def __rsub__(self, other):
        return self._add(other, subtract=True, reverse=True)

    def __neg__(self):
        return Fixed(-self.raw, self.bits + 1, self.frac, True)

    def __abs__(self):
        return Fixed(np.abs(self.raw), self.bits, self.frac, False)

    def __mul__(self, other):
        other = self._coerce(other, 0)
        signed = self.signed or other.signed
        bits = self._check_sim(self.bits + other.bits, self.required_bits() + other.required_bits())
        return Fixed(self.raw * other.raw, bits, self.frac + other.frac, signed)

    __rmul__ = __mul__

    def __matmul__(self, other):
        """Multiply-accumulate over the last axis of self, like a MAC array."""
        if not isinstance(other, Fixed):
            raise TypeError("matmul needs two Fixed operands")
        terms = _clog2(self.shape[-1])
        bits = self._check_sim(
            self.bits + other.bits + terms,
            self.required_bits() + other.required_bits() + terms,
        )
        return Fixed(self.raw @ other.raw, bits, self.frac + other.frac, True)

    def sum(self, axis=None, keepdims=False):
        count = self.raw.size if axis is None else self.raw.shape[axis]
        bits = self._check_sim(self.bits + _clog2(count), self.required_bits() + _clog2(count))
        return Fixed(self.raw.sum(axis=axis, keepdims=keepdims), bits, self.frac, self.signed)

    def max(self, axis=None, keepdims=False, where=None):
        """Comparator tree; masked-out elements are ignored (empty rows give the format minimum)."""
        if where is None:
            return self._like(self.raw.max(axis=axis, keepdims=keepdims))
        low = format_range(self.bits, self.signed)[0]
        return self._like(self.raw.max(axis=axis, keepdims=keepdims, where=where, initial=low))

    def min(self, axis=None, keepdims=False):
        return self._like(self.raw.min(axis=axis, keepdims=keepdims))

    def clip(self, low=None, high=None):
        """Saturate to [low, high] (raw values); the width shrinks to fit the bounds."""
        current_low, current_high = format_range(self.bits, self.signed)
        low = current_low if low is None else max(int(low), current_low)
        high = current_high if high is None else min(int(high), current_high)
        return Fixed(np.clip(self.raw, low, high), bits_for(low, high, self.signed),
                     self.frac, self.signed)

    # ----- shifts -----

    def __rshift__(self, shift):
        """Arithmetic right shift of the raw bits (floor); the fraction is unchanged."""
        low, _ = _shift_range(shift)
        if low < 0:
            raise ValueError("Negative right shift; use floor_shift")
        return Fixed(self.raw >> _raw(shift), max(self.bits - low, 1), self.frac, self.signed)

    def __lshift__(self, shift):
        """Left shift of the raw bits; the fraction is unchanged."""
        low, high = _shift_range(shift)
        if low < 0:
            raise ValueError("Negative left shift; use floor_shift")
        bits = self._check_sim(self.bits + high, self.required_bits() + high)
        return Fixed(self.raw << _raw(shift), bits, self.frac, self.signed)

    def __and__(self, mask):
        """AND with a non-negative literal mask: keeps the low bits, result unsigned."""
        if isinstance(mask, Fixed) or int(mask) < 0:
            raise TypeError("Only a non-negative integer literal mask is supported")
        return Fixed(self.raw & int(mask), max(int(mask).bit_length(), 1), self.frac, False)

    def _shift_frac(self, shift, frac):
        if frac is not None:
            return frac
        if isinstance(shift, (int, np.integer)):
            return self.frac - int(shift)
        raise ValueError("A per-element shift needs the result fraction: pass frac=")

    def floor_shift(self, shift, frac=None):
        """Shift right by `shift` (negative = left), rounding toward -inf.

        A scalar shift moves the binary point (frac -= shift) so the value is
        kept; a per-element shift needs the result fraction explicitly.
        """
        low, high = _shift_range(shift)
        bits = self._check_sim(max(self.bits - low, 1), max(self.required_bits() - low, 1))
        raw_shift = _raw(shift)
        right = np.maximum(raw_shift, 0)
        raw = (self.raw >> right) << np.maximum(-raw_shift, 0)
        return Fixed(raw, bits, self._shift_frac(shift, frac), self.signed)

    def round_shift(self, shift, frac=None):
        """Shift right rounding half up (negative = left); see floor_shift for frac."""
        low, high = _shift_range(shift)
        grow = 1 if low > 0 else 0
        bits = self._check_sim(max(self.bits - low + grow, 1),
                               max(self.required_bits() - low + grow, 1))
        raw_shift = _raw(shift)
        right = np.maximum(raw_shift, 0)
        half = np.where(right > 0, np.int64(1) << np.maximum(right - 1, 0), 0)
        raw = ((self.raw + half) >> right) << np.maximum(-raw_shift, 0)
        return Fixed(raw, bits, self._shift_frac(shift, frac), self.signed)

    # ----- comparisons (return np.bool_ arrays) -----

    def _compare_raw(self, other):
        other = self._coerce(other, self.frac)
        if other.frac != self.frac:
            raise ValueError(f"Align fractions before comparing: Q.{self.frac} and Q.{other.frac}")
        return other.raw

    def __lt__(self, other):
        return self.raw < self._compare_raw(other)

    def __le__(self, other):
        return self.raw <= self._compare_raw(other)

    def __gt__(self, other):
        return self.raw > self._compare_raw(other)

    def __ge__(self, other):
        return self.raw >= self._compare_raw(other)

    def __eq__(self, other):
        return self.raw == self._compare_raw(other)

    def __ne__(self, other):
        return self.raw != self._compare_raw(other)

    __hash__ = None


def where(condition, a, b):
    """Multiplexer: a where condition holds, else b."""
    frac = a.frac if isinstance(a, Fixed) else b.frac if isinstance(b, Fixed) else 0
    a, b = (x if isinstance(x, Fixed) else Fixed.const(x, frac) for x in (a, b))
    if a.frac != b.frac:
        raise ValueError("Multiplexer inputs must share a fraction")
    signed = a.signed or b.signed
    if signed:
        bits = max(a._signed_bits(), b._signed_bits())
    else:
        bits = max(a.bits, b.bits)
    return Fixed(np.where(condition, a.raw, b.raw), bits, a.frac, signed)


def leading_one(x):
    """Priority encoder: bits needed to represent x >= 0 (0 for x = 0)."""
    if (x.raw < 0).any():
        raise ValueError("leading_one needs non-negative values")
    length = np.zeros_like(x.raw)
    for bit in range(x.bits):
        length = np.where(x.raw >> bit != 0, bit + 1, length)
    return Fixed(length, max(x.bits.bit_length(), 1), 0, False)


def fixed_type(name, bits, frac=0, signed=True):
    """Make a named register type, e.g. Int8 = fixed_type("Int8", 8)."""

    def __init__(self, raw, overflow="error", name=None):
        Fixed.__init__(self, raw, bits, frac, signed, overflow, name)

    @classmethod
    def from_float(cls, value, overflow="error", name=None):
        raw = np.rint(np.asarray(value, dtype=np.float64) * 2.0**frac).astype(np.int64)
        return cls(raw, overflow, name)

    doc = f"{'Signed' if signed else 'Unsigned'} {bits}-bit register, {frac} fraction bits."
    return type(name, (Fixed,), {
        "__init__": __init__, "from_float": from_float, "__slots__": (), "__doc__": doc,
        "BITS": bits, "FRAC": frac, "SIGNED": signed,
    })


Int8 = fixed_type("Int8", 8)
Int16 = fixed_type("Int16", 16)
Int32 = fixed_type("Int32", 32)
UInt8 = fixed_type("UInt8", 8, signed=False)
# The residual stream and every wide tensor: Q.16 in a 32-bit register.
Q16 = fixed_type("Q16", 32, frac=16)
