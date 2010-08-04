"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_get_random_random():
    """
    Test construction of randomly-initialized PRNGs.
    """

    import numpy.random

    from nose.tools   import assert_true
    from cargo.random import get_random_random

    random = get_random_random(numpy.random)

    assert_true(0 <= random.randint(16) < 16)

def test_grab():
    """
    Test random element selection using grab().
    """

    from nose.tools   import assert_almost_equal
    from numpy.random import RandomState
    from cargo.random import grab

    random   = RandomState(42)
    sequence = range(2)
    grabbed  = [grab(sequence, random) for i in xrange(4096)]

    assert_almost_equal(sum(grabbed) / 4096.0, 0.5, places = 2)

