"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from nose.tools import assert_equal

def test_type_from_dtype_complex():
    """
    Test to-LLVM translation of a complex numpy dtype.
    """

    from cargo.llvm import type_from_dtype

    dtype = numpy.dtype([("d", [("k", numpy.uint32), ("n", numpy.uint32)], (4,))])
    type_ = type_from_dtype(dtype)

    assert_equal(str(type_), "<{ [4 x <{ i32, i32 }>] }>")

def test_dtype_from_type_complex():
    """
    Test to-numpy translation of a complex LLVM type.
    """

    from llvm.core  import Type
    from cargo.llvm import (
        type_from_dtype,
        dtype_from_type,
        )

    dtype = numpy.dtype([("f0", [("f0", numpy.int32), ("f1", numpy.int32)], (4,))])
    type_ = Type.struct([Type.array(Type.packed_struct([Type.int(32)] * 2), 4)])
    dtype2 = dtype_from_type(type_)

    assert_equal(dtype2.itemsize, dtype.itemsize)
    assert_equal(str(dtype2), str(dtype))

