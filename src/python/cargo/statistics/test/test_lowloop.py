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

    # compile array copy via array loop
    from cargo.statistics.lowloop import strided_array_loop

    local   = Module.new("local")
    main    = local.add_function(Type.function(Type.void(), []), "main")
    entry   = main.append_basic_block("entry")
    builder = Builder.new(entry)

    def emit_loop_copy(builder, locations):
        """
        Copy a single element.
        """

        builder.store(
            builder.load(locations["in"]),
            locations["out"],
            )

    strided_array_loop(
        builder,
        emit_loop_copy,
        out.shape,
        {"in" : in_, "out" : out},
        )

    builder.ret_void()

    # execute the loop
    local.verify()

    engine = ExecutionEngine.new(local)

    engine.run_function(main, [])

    # verify correctness
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

