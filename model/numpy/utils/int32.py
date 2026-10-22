from .int8 import _SignedInt


class Int32(_SignedInt):
    """Signed 32-bit scalar, wrapping to [-2147483648, 2147483647].

    Cast INT8 operands to Int32 before multiplication for a wide result.
    """

    BITS = 32
    MIN = -2147483648
    MAX = 2147483647
