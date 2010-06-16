"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_multinomial_log_probability():
    """
    Test computation of multinomial log-probability.
    """

    import numpy

    from nose.tools                   import assert_equal
    from cargo.statistics.multinomial import Multinomial

    m  = Multinomial([0.5, 0.5])
    lp = m.log_likelihood(numpy.array([1, 0], dtype = numpy.uint))
    
    assert_equal(numpy.exp(lp), 0.5)

    lp = m.log_likelihood(numpy.array([0, 2], dtype = numpy.uint))

    assert_equal(numpy.exp(lp), 0.25)

