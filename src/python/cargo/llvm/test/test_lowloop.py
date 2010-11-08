"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from nose.tools import assert_equal
from llvm.core  import (
    Type,
    Module,
    Builder,
    )
from llvm.ee    import ExecutionEngine
from cargo.llvm import (
    high,
    emit_and_execute,
    StridedArray,
    StridedArrays,
    )

def assert_copying_ok(in_, out, expected):
    """
    Assert that the array loop can make a correct array copy.
    """

    @emit_and_execute()
    def _():
        arrays = StridedArrays.from_numpy({"in" : in_, "out" : out})

        @arrays.loop_all()
        def _(l):
            l.arrays["in"].data.load().store(l.arrays["out"].data)

    assert_equal(expected.tolist(), out.tolist())

def test_strided_arrays_loop_all_simple():
    """
    Test basic strided-array loop compilation.
    """

    # generate some test data
    foo = numpy.random.randint(10, size = (2, 4))
    bar = numpy.random.randint(10, size = (2, 4))

    # verify correctness
    assert_copying_ok(foo, bar, numpy.copy(foo))

def test_strided_arrays_loop_all_broadcast():
    """
    Test strided-array loop compilation with broadcast arrays.
    """

    # generate some test data
    (foo, bar) = \
        numpy.broadcast_arrays(
            numpy.random.randint(10, size = (2, 1, 6)),
            numpy.random.randint(10, size = (2, 4, 6)),
            )
    baz = numpy.empty(bar.shape, numpy.int)

    baz[:] = foo

    # verify correctness
    assert_copying_ok(foo, bar, baz)

def test_strided_arrays_loop_all_views():
    """
    Test strided-array loop compilation on numpy views.
    """

    # generate some test data
    (foo, bar) = \
        numpy.broadcast_arrays(
            numpy.random.randint(10, size = (10, 10))[2:8:2, 5: 6  ],
            numpy.random.randint(10, size = (10, 10))[2:8:2, 1:10:3],
            )
    baz = numpy.empty(bar.shape, numpy.int)

    baz[:] = foo

    # verify correctness
    assert_copying_ok(foo, bar, baz)

def test_strided_arrays_loop_all_subarrays():
    """
    Test strided-array loop compilation on subarrays.
    """

    # generate some test data
    (foo, bar) = \
        numpy.broadcast_arrays(
            numpy.random.randint(10, size = (2, 2, 4, 6)),
            numpy.random.randint(10, size = (2, 2, 4, 6)),
            )
    baz = numpy.empty(bar.shape[1:], numpy.int)

    baz[:] = foo[0]

    # verify correctness
    @emit_and_execute()
    def _():
        arrays = \
            StridedArrays({
                "in"  : StridedArray.from_numpy(foo).at(0),
                "out" : StridedArray.from_numpy(bar).at(1),
                })

        @arrays.loop_all()
        def _(l):
            l.arrays["in"].data.load().store(l.arrays["out"].data)

    assert_equal(bar[1].tolist(), baz.tolist())

def test_strided_array_extract():
    """
    Test strided-array loop compilation on extracted member arrays.
    """

    # generate some test data
    foo = numpy.empty((2, 4), [("x", int), ("y", float)])
    bar = numpy.empty((2, 4))
    baz = numpy.random.rand(2, 4)

    foo["y"] = baz

    # verify correctness
    @emit_and_execute()
    def _():
        arrays = \
            StridedArrays({
                "in"  : StridedArray.from_numpy(foo).extract(0, 1),
                "out" : StridedArray.from_numpy(bar),
                })

        @arrays.loop_all()
        def _(l):
            l.arrays["in"].data.load().store(l.arrays["out"].data)

    assert_equal(bar.tolist(), baz.tolist())

def test_strided_array_at_complex_dtype():
    """
    Test strided-array addressing with a complex dtype.
    """

    # generate some test data
    dtype = numpy.dtype([("d", [("k", numpy.uint32), ("n", numpy.uint32)], (4,))])
    array = numpy.empty(8, dtype)

    array["d"]["k"] = 0
    array["d"]["n"] = 42

    # verify correctness
    from cargo.llvm import iptr_type

    @emit_and_execute()
    def _():
        ll_array = StridedArray.from_numpy(array)

        at0 = ll_array.at(0).data.cast_to(iptr_type)
        at1 = ll_array.at(1).data.cast_to(iptr_type)

        @high.python(at0, at1)
        def _(at0_py, at1_py):
            assert_equal(at0_py, array[0].__array_interface__["data"][0])
            assert_equal(at1_py, array[1].__array_interface__["data"][0])

def test_strided_array_loop_subshape_complex_dtype():
    """
    Test strided-array loop compilation with a complex dtype.
    """

    # generate some test data
    dtype = numpy.dtype([("d", [("k", numpy.uint32), ("n", numpy.uint32)], (4,))])
    array = numpy.empty(2, dtype)

    array["d"]["k"] = 0
    array["d"]["n"] = 42

    # verify correctness
    from cargo.llvm import iptr_type

    @emit_and_execute()
    def _():
        ll_array  = StridedArray.from_numpy(array)
        ll_arrays = StridedArrays({"a" : ll_array})

        @ll_arrays.loop_all()
        def _(l):
            high.py_printf("address %i", l.arrays["a"].data.cast_to(iptr_type))
            #@high.python(at0, at1)
            #def _(at0_py, at1_py):
                #assert_equal(at0_py, array[0].__array_interface__["data"][0])
                #assert_equal(at1_py, array[1].__array_interface__["data"][0])

