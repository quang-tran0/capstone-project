import operator
from dataclasses import dataclass

from .int8 import Int8
from .scale import quantize_scale


@dataclass(frozen=True, init=False)
class ScaleInt12:
    """Scale in [0, 127], approximated by multiplier * 2**(-degree).

    Packed layout: multiplier in bits [11:4], unsigned degree in bits [3:0].
    Small scales may round to zero, as in quantize_scale. multiply_wide()
    retains wider fields until the output scale is computed.
    """

    multiplier: Int8 | int
    degree: int

    def __init__(self, value: float):
        multiplier, degree = quantize_scale(value)
        object.__setattr__(self, "multiplier", Int8(multiplier))
        object.__setattr__(self, "degree", int(degree))

    @property
    def value(self) -> float:
        return int(self.multiplier) * 2.0**(-self.degree)

    def __float__(self):
        return self.value

    @classmethod
    def from_parts(cls, multiplier, degree, *, normalize=True):
        """Construct m|n, optionally retaining wide intermediate fields."""
        multiplier, degree = operator.index(multiplier), operator.index(degree)
        if multiplier < 0 or degree < 0:
            raise ValueError("Scale fields must be nonnegative")
        if normalize:
            if multiplier > (127 << degree):
                raise OverflowError("Scale exceeds the 12-bit representable maximum 127")
            return cls._normalize_parts(multiplier, degree)
        result = object.__new__(cls)
        object.__setattr__(result, "multiplier", multiplier)
        object.__setattr__(result, "degree", degree)
        return result

    def normalize(self):
        """Return a 12-bit scale with the largest fitting multiplier.

        Double the multiplier and increment the degree while the doubled
        multiplier fits INT8 and the degree fits UINT4. Wide fields may
        require rounded right shifts first. Zero uses degree 15.
        """
        return type(self).from_parts(int(self.multiplier), self.degree)

    @classmethod
    def _normalize_parts(cls, multiplier, degree):
        # Keep the intermediate product wide until it fits both fields.
        shift = max(0, multiplier.bit_length() - 7, degree - 15)
        if shift:
            while True:
                rounded, remainder = divmod(multiplier, 1 << shift)
                halfway = 1 << (shift - 1)
                if remainder > halfway or (remainder == halfway and rounded & 1):
                    rounded += 1
                if rounded <= 127:
                    break
                shift += 1
            multiplier = rounded
            degree -= shift
        if multiplier == 0:
            degree = 15
        else:
            while multiplier * 2 < 128 and degree < 15:
                multiplier *= 2
                degree += 1
        return cls.unpack((multiplier << 4) | degree)

    def __mul__(self, other):
        """Multiply multipliers, add degrees, then normalize to 12 bits.

        Right shifts reduce the degree by the same amount and round to
        nearest, ties to even. Values below the resolution may become zero.
        """
        if not isinstance(other, ScaleInt12):
            return NotImplemented
        degree = self.degree + other.degree
        multiplier = int(self.multiplier) * int(other.multiplier)
        return type(self).from_parts(multiplier, degree)

    def multiply_wide(self, other):
        """Keep exact product fields, without INT8/UINT4 rounding.

        An accumulator scale S_X*S_W must not become zero before the
        accumulator peak is included in the output scale calculation.
        """
        if not isinstance(other, ScaleInt12):
            raise TypeError("Scale product requires another ScaleInt12")
        return type(self).from_parts(
            int(self.multiplier) * int(other.multiplier),
            self.degree + other.degree,
            normalize=False,
        )

    def pack(self) -> int:
        """Return the unsigned 12-bit encoding, without changing its fields."""
        if not (0 <= int(self.multiplier) <= 127 and 0 <= self.degree <= 15):
            raise OverflowError("Intermediate scale must be normalized before packing")
        return ((int(self.multiplier) & 0xFF) << 4) | self.degree

    @classmethod
    def unpack(cls, packed: int):
        """Decode valid nonnegative fields exactly, without normalizing."""
        packed = operator.index(packed)
        if not 0 <= packed <= 0xFFF:
            raise ValueError("Packed scale must fit in 12 unsigned bits")
        multiplier = Int8(packed >> 4)
        degree = packed & 0xF
        if multiplier < 0:
            raise ValueError("Decoded scale must be nonnegative")
        result = object.__new__(cls)
        object.__setattr__(result, "multiplier", multiplier)
        object.__setattr__(result, "degree", degree)
        return result
