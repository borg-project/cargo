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

def assert_copying_ok(in_, out, expected):
    """
    Assert that the array loop can make a correct array copy.
    """

    from cargo.llvm import (
        emit_and_execute,
        StridedArrays,
        )

    @emit_and_execute()
    def _(_):
        arrays = StridedArrays.from_numpy({"in" : in_, "out" : out})

        @arrays.loop_all()
        def _(l):
            l.arrays["in"].data.load().store(l.arrays["out"].data)

    assert_equal(expected.tolist(), out.tolist())

def test_array_loop_simple():
    """
    Test basic strided-array loop compilation.
    """

    # generate some test data
    foo = numpy.random.randint(10, size = (2, 4))
    bar = numpy.random.randint(10, size = (2, 4))

    # verify correctness
    assert_copying_ok(foo, bar, numpy.copy(foo))

def test_array_loop_broadcast():
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

def test_array_loop_views():
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

def test_array_loop_subarrays():
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
    from cargo.llvm import (
        emit_and_execute,
        StridedArray,
        StridedArrays,
        )

    @emit_and_execute()
    def _(_):
        arrays = \
            StridedArrays({
                "in"  : StridedArray.from_numpy(foo).at(0),
                "out" : StridedArray.from_numpy(bar).at(1),
                })

        @arrays.loop_all()
        def _(l):
            l.arrays["in"].data.load().store(l.arrays["out"].data)

    assert_equal(bar[1].tolist(), baz.tolist())

