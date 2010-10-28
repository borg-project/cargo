"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy

from cargo.llvm import HighFunction

cdef double random_double():
    """
    Emit a PRNG invocation.
    """

    return numpy.random.rand()

cdef long random_int(long upper):
    """
    Emit a PRNG invocation.
    """

    return numpy.random.randint(upper)

def emit_random_real_unit(high):
    """
    Emit a PRNG invocation.
    """

    c_random_double = HighFunction(<long>&random_double, float, [])

    return c_random_double()

def emit_random_int(high, upper, width):
    """
    Emit a PRNG invocation.
    """

    c_random_int = HighFunction(<long>&random_int, ctypes.c_long, [ctypes.c_long])

    return c_random_int(upper)

