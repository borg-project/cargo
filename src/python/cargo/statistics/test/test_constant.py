"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_constant():
    """
    Test the trivial constant distribution.
    """

    import numpy

    from nose.tools                    import assert_equal
    from cargo.statistics.distribution import Constant

    constant = Constant(42.0)

    assert_equal(constant.random_variate(), 42.0)
    assert_equal(constant.log_likelihood(42.1), numpy.finfo(float).min)
    assert_equal(constant.log_likelihood("42"), numpy.finfo(float).min)
    assert_equal(constant.log_likelihood(42.0), 0.0)

