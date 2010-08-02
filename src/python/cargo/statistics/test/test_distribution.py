"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_constant_distribution():
    """
    Test the trivial constant distribution.
    """

    import numpy

    from nose.tools                    import assert_equal
    from cargo.statistics.distribution import ConstantDistribution

    distribution = ConstantDistribution(42.0)

    assert_equal(distribution.random_variate(), 42.0)
    assert_equal(distribution.log_likelihood(42.1), numpy.finfo(float).min)
    assert_equal(distribution.log_likelihood("42"), numpy.finfo(float).min)
    assert_equal(distribution.log_likelihood(42.0), 0.0)

