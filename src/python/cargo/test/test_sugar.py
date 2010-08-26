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

