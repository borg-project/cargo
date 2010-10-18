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

def test_array_loop_simple():
    """
    Test basic strided-array loop compilation.
    """

    # generate some test data
    shape  = (2, 4)
    arrays = {
        "foo" : numpy.random.randint(10, size = shape),
        "bar" : numpy.random.randint(10, size = shape),
        }

    # compile summation via array loop
    from cargo.statistics.lowloop import ArrayLoop

    local = Module.new("local")
    main  = local.add_function(Type.function(Type.void(), []), "main")
    entry = main.append_basic_block("entry")
    exit  = main.append_basic_block("exit")
    loop  = ArrayLoop(main, shape, exit, arrays)

    Builder.new(entry).branch(loop.entry)
    Builder.new(exit).ret_void()

    loop.builder.store(
        loop.builder.load(loop.locations["foo"]),
        loop.locations["bar"],
        )

    local.verify()

    # execute the loop
    engine = ExecutionEngine.new(local)
    total  = engine.run_function(main, [])

    # verify correctness
    assert_equal(arrays["foo"].tolist(), arrays["bar"].tolist())

