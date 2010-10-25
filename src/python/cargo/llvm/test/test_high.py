"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    assert_true,
    assert_equal,
    )
from llvm.core  import (
    Type,
    Module,
    Builder,
    )

def emit_and_execute(emit_test_body):
    """
    Prepare for, emit, and run some LLVM IR.
    """

    # emit some IR
    from cargo.llvm import this_builder

    local = Module.new("local")
    main  = local.add_function(Type.function(Type.void(), []), "main")
    entry = main.append_basic_block("entry")

    with this_builder(Builder.new(entry)) as builder:
        emit_test_body()

        builder.ret_void()

    # then compile and execute it
    from llvm.ee import ExecutionEngine

    local.verify()

    engine = ExecutionEngine.new(local)

    engine.run_function(main, [])

def test_high_python():
    """
    Test the high-LLVM python() construct.
    """

    from cargo.llvm.high_level import high

    executed = [False]

    @emit_and_execute
    def run_test():
        @high.python()
        def print_something():
            executed[0] = [True]

    assert_true(executed[0])

def test_high_for_():
    """
    Test the high-LLVM for_() loop construct.
    """

    from cargo.llvm.high_level import high

    count      = 128
    iterations = [0]

    @emit_and_execute
    def run_test():
        @high.for_(count)
        def _(_):
            @high.python()
            def _():
                iterations[0] += 1

    assert_equal(iterations[0], count)

def test_high_nested_for_():
    """
    Test the high-LLVM for_() loop construct, nested.
    """

    from cargo.llvm.high_level import high

    count      = 32
    iterations = [0]

    @emit_and_execute
    def run_test():
        @high.for_(count)
        def _(_):
            @high.for_(count)
            def _(_):
                @high.python()
                def _():
                    iterations[0] += 1

    assert_equal(iterations[0], count**2)

