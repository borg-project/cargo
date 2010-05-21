"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_get_random_random():
    """
    Test construction of randomly-initialized PRNGs.
    """

    import numpy.random

    from nose.tools import assert_true

    random = get_random_random(numpy.random)

    assert_true(0 <= random.randint(16) < 16)

