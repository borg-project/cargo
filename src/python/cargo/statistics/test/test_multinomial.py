"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_multinomial():
    """
    Test the multinomial distribution.
    """

    # build a simple distribution
    import numpy

    from cargo.statistics.multinomial import Multinomial

    m = Multinomial([0.25, 0.75], 1)

    # test ll computation
    def test_log_probability():
        """
        Test computation of multinomial log probability.
        """

        from nose.tools import assert_equal

        lp = m.log_likelihood(numpy.array([1, 0], numpy.uint))
        
        assert_equal(numpy.exp(lp), 0.25)

        # FIXME doesn't truly make sense, given the distribution's norm
        lp = m.log_likelihood(numpy.array([0, 2], numpy.uint))

        assert_equal(numpy.exp(lp), 0.5625)

    yield test_log_probability

    # test sampling
    def test_random_variate():
        """
        Test multinomial random variate generation.
        """

        from nose.tools import assert_almost_equal

        totals = numpy.zeros(2, numpy.uint)

        for i in xrange(4096):
            totals += m.random_variate()

        assert_almost_equal(totals[0] / 4096.0, 0.25, places = 2)
        assert_almost_equal(totals[1] / 4096.0, 0.75, places = 2)

    yield test_random_variate

def test_multinomial_estimator():
    """
    Test estimation of the multinomial distribution.
    """

    raise NotImplementedError()

