import operator


class _SignedInt(int):
    """Scalar signed integer with two's-complement wrapping.

    Integer arithmetic and bit operations wrap to the operand's width.
    Mixed fixed-width operands use the wider width; plain integers use
    this operand's width. Comparisons and true division follow Python int.
    """

    def __new__(cls, value=0):
        value = operator.index(value)
        value = (value - cls.MIN) % (1 << cls.BITS) + cls.MIN
        return super().__new__(cls, value)

    @property
    def value(self):
        return int(self)

    def __repr__(self):
        return f"{type(self).__name__}({int(self)})"

    def _binary(self, other, operation, reverse=False):
        try:
            other_value = operator.index(other)
        except TypeError:
            return NotImplemented
        result_type = type(self)
        if isinstance(other, _SignedInt) and other.BITS > self.BITS:
            result_type = type(other)
        left, right = int(self), other_value
        if reverse:
            left, right = right, left
        return result_type(operation(left, right))

    def __add__(self, other):
        return self._binary(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        return self._binary(other, operator.sub, reverse=True)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return self._binary(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._binary(other, operator.floordiv, reverse=True)

    def __mod__(self, other):
        return self._binary(other, operator.mod)

    def __rmod__(self, other):
        return self._binary(other, operator.mod, reverse=True)

    def __and__(self, other):
        return self._binary(other, operator.and_)

    __rand__ = __and__

    def __or__(self, other):
        return self._binary(other, operator.or_)

    __ror__ = __or__

    def __xor__(self, other):
        return self._binary(other, operator.xor)

    __rxor__ = __xor__

    def __lshift__(self, other):
        return type(self)(int(self) << operator.index(other))

    def __rlshift__(self, other):
        return type(self)(operator.index(other) << int(self))

    def __rshift__(self, other):
        return type(self)(int(self) >> operator.index(other))

    def __rrshift__(self, other):
        return type(self)(operator.index(other) >> int(self))

    def __neg__(self):
        return type(self)(-int(self))

    def __pos__(self):
        return self

    def __abs__(self):
        return type(self)(abs(int(self)))

    def __invert__(self):
        return type(self)(~int(self))


class Int8(_SignedInt):
    """Signed 8-bit scalar, wrapping to [-128, 127]."""

    BITS = 8
    MIN = -128
    MAX = 127
