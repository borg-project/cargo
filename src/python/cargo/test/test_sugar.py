"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_value_by_name():
    """
    Test value retrieval by name.
    """

    from nose.tools  import assert_equal
    from cargo.sugar import value_by_name

    value = value_by_name("cargo.test.test_sugar.test_value_by_name")

    assert_equal(value, test_value_by_name)

def test_composed():
    """
    Test the function composition decorator.
    """

    from nose.tools  import (
        assert_true,
        assert_equal,
        )
    from cargo.sugar import composed

    @composed(list)
    def range_four():
        for i in xrange(4):
            yield i

    assert_true(isinstance(range_four(), list))
    assert_equal(range_four(), range(4))

