"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes

from llvm.core import Type

iptr_type = Type.int(ctypes.sizeof(ctypes.c_void_p) * 8)

def for_(function, exit, low_or_high, high = None, name = "for"):
    """
    Generate a simple ranged for loop.
    """

    if high is None:
        low  = 0
        high = low_or_high
    else:
        low = low_or_high

def constant_stringz_p(string):
    """
    Return a pointer to the beginning of a constant string.
    """

