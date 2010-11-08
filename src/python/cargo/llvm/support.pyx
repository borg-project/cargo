"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy

from cargo.llvm import HighFunction

from cpython.exc cimport PyErr_Occurred

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

    c_random_double = HighFunction.pointed(<long>&random_double, float, [])

    return c_random_double()

def emit_random_int(high, upper, width):
    """
    Emit a PRNG invocation.
    """

    c_random_int = HighFunction.pointed(<long>&random_int, ctypes.c_long, [ctypes.c_long])

    return c_random_int(upper)

cdef extern from "setjmp.h":
    struct __jmp_buf_tag:
        # GNU-specific (?) jump buffer type.
        pass

def size_of_jmp_buf():
    """
    Return the size of a jmp_buf structure in bytes.
    """

    return sizeof(__jmp_buf_tag)

cpdef int raise_if_set() except 1:
    """
    Force the Python runtime to notice an exception, if one is set.
    """

    if PyErr_Occurred():
        return 1
    else:
        return 0

