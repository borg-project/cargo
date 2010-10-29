"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    )
from cargo.llvm import (
    high,
    emit_and_execute,
    )

def test_high_python_no_arguments():
    """
    Test the python() LLVM construct without arguments.
    """

    executed = [False]

    @emit_and_execute()
    def _(_):
        @high.python()
        def _():
            executed[0] = [True]

    assert_true(executed[0])

def test_high_python_arguments():
    """
    Test the python() LLVM construct with arguments.
    """

    values = []

    @emit_and_execute()
    def _(_):
        @high.for_(8)
        def _(i):
            @high.python(i)
            def _(j):
                values.append(j)

    assert_equal(values, range(8))

def test_high_for_():
    """
    Test the high-LLVM for_() loop construct.
    """

    count      = 128
    iterations = [0]

    @emit_and_execute()
    def _(_):
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

    count      = 32
    iterations = [0]

    @emit_and_execute()
    def _(_):
        @high.for_(count)
        def _(_):
            @high.for_(count)
            def _(_):
                @high.python()
                def _():
                    iterations[0] += 1

    assert_equal(iterations[0], count**2)

def test_high_random():
    """
    Test the high-LLVM random() construct.
    """

    count = 4096
    total = [0.0]

    @emit_and_execute()
    def _(_):
        @high.for_(count)
        def _(_):
            v = high.random()

            @high.python(v)
            def _(v_py):
                total[0] += v_py

    assert_almost_equal(total[0] / count, 0.5, places = 1)

def test_high_random_int():
    """
    Test the high-LLVM random_int() construct.
    """

    count  = 32
    values = []

    @emit_and_execute()
    def _(_):
        @high.for_(count)
        def _(_):
            v = high.random_int(2)

            @high.python(v)
            def _(v_py):
                values.append(v_py)

    assert_true(len(filter(None, values)) > 8)
    assert_true(len(filter(None, values)) < 24)

def test_high_select():
    """
    Test the select() LLVM construct without arguments.
    """

    result = [None, None]

    @emit_and_execute()
    def _(_):
        v0 = high.select(True, 3, 4)
        v1 = high.select(False, 3, 4)

        @high.python(v0, v1)
        def _(v0_py, v1_py):
            result[0] = v0_py
            result[1] = v1_py

    assert_equal(result[0], 3)
    assert_equal(result[1], 4)

