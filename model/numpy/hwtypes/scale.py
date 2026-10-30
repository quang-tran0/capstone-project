"""Power-of-two scales m * 2**-n with configurable field widths."""

from dataclasses import dataclass

import numpy as np

from .fixed import Fixed


# 1/code_max in Q.24, rounded up so a scale never falls below peak / code_max.
RECIP_CODE_BITS = 24


@dataclass(frozen=True)
class ScaleFormat:
    """Widths of the two scale fields: unsigned m (m_bits) and unsigned n (n_bits).

    The default 7 + 5 bits gives m in [1, 127] and n in [0, 31].
    """

    m_bits: int = 7
    n_bits: int = 5

    @property
    def m_max(self):
        return (1 << self.m_bits) - 1

    @property
    def n_max(self):
        return (1 << self.n_bits) - 1

    def scale(self, m, n):
        """Wrap raw field values, checking they fit this format."""
        return Scale(Fixed(m, self.m_bits, signed=False, name="scale m"),
                     Fixed(n, self.n_bits, signed=False, name="scale n"))

    def from_float(self, scale):
        """Offline: largest n whose m still fits, with m rounded up (never below scale)."""
        scale = np.maximum(np.asarray(scale, dtype=np.float64), 2.0**-self.n_max)
        degree = np.clip(np.floor(np.log2(self.m_max / scale)), 0, self.n_max)
        degree = np.where(np.ceil(scale * 2.0**degree) > self.m_max, degree - 1, degree)
        multiplier = np.maximum(np.ceil(scale * 2.0**degree), 1)
        return self.scale(multiplier.astype(np.int64), degree.astype(np.int64))

    def from_peak(self, peak, code_max):
        """Runtime, integer-only: m * 2**-n ~= peak / code_max, m rounded up.

        peak is a non-negative Fixed; its fraction is honoured. All n_max + 1
        candidates are formed with ceiling shifts and the largest n whose m
        fits is picked, like a priority encoder. A zero peak gets m = 1, n = n_max.
        """
        recip = Fixed.const(-(-(1 << RECIP_CODE_BITS) // code_max))
        scaled = (peak * recip).raw

        shifts = peak.frac + RECIP_CODE_BITS - np.arange(self.n_max + 1)
        if shifts.min() < 0:
            raise ValueError("n_max exceeds the precision of the peak / code_max product")
        # Ceiling shift: (v + 2**k - 1) >> k.
        candidates = (scaled[..., None] + (np.int64(1) << shifts) - 1) >> shifts
        fits = candidates <= self.m_max
        if not fits[..., 0].all():
            raise OverflowError("Peak too large for the m field at n = 0")

        degree = fits.sum(axis=-1) - 1
        multiplier = np.take_along_axis(candidates, degree[..., None], axis=-1)[..., 0]
        zero = multiplier == 0
        return self.scale(np.where(zero, 1, multiplier), np.where(zero, self.n_max, degree))


class Scale:
    """Scale m * 2**-n stored as two unsigned Fixed fields (broadcastable arrays).

    Products keep the exact fields (m multiplies, n adds), so a product scale
    is wider than either input, as in the hardware multiplier.
    """

    __slots__ = ("m", "n")

    def __init__(self, m, n):
        if not isinstance(m, Fixed) or not isinstance(n, Fixed):
            raise TypeError("Scale fields must be Fixed")
        if m.frac or n.frac:
            raise ValueError("Scale fields are plain integers (frac 0)")
        self.m = m
        self.n = n

    @classmethod
    def const(cls, m, n):
        """A literal scale, e.g. 1/sqrt(32) ~= Scale.const(5793, 15)."""
        return cls(Fixed.const(m), Fixed.const(n))

    def __repr__(self):
        return f"Scale(m={self.m!r}, n={self.n!r})"

    @property
    def shape(self):
        return np.broadcast_shapes(self.m.shape, self.n.shape)

    def __mul__(self, other):
        if not isinstance(other, Scale):
            return NotImplemented
        return Scale(self.m * other.m, self.n + other.n)

    def __getitem__(self, key):
        return Scale(self.m[key], self.n[key])

    def swapaxes(self, axis1, axis2):
        return Scale(self.m.swapaxes(axis1, axis2), self.n.swapaxes(axis1, axis2))

    def to_float(self):
        """Debug/trace only."""
        return self.m.raw * np.exp2(-self.n.raw.astype(np.float64))

    def pack(self):
        """{m, n} concatenated into one unsigned word of m.bits + n.bits."""
        if self.m.signed or self.n.signed:
            raise ValueError("Only unsigned fields can be packed")
        return Fixed((self.m.raw << self.n.bits) | self.n.raw,
                     self.m.bits + self.n.bits, signed=False)

    def rescale(self, value, frac):
        """value * m * 2**-n as a Q.frac Fixed: one multiply, one rounding shift."""
        return (value * self.m).round_shift(self.n + (value.frac - frac), frac=frac)


@dataclass
class QTensor:
    """Integer codes with their scale: value ~= codes * m * 2**-n."""

    codes: Fixed
    scale: Scale

    def dequantize(self):
        """Debug/trace only."""
        return self.codes.to_float() * self.scale.to_float()
