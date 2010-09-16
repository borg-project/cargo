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

        from nose.tools import assert_almost_equal

        lp = m.log_likelihood(numpy.array([1, 1], numpy.uint))

        assert_almost_equal(numpy.exp(lp), 0.375)

        lp = m.log_likelihood(numpy.array([0, 2], numpy.uint))

        assert_almost_equal(numpy.exp(lp), 0.5625)

    yield test_log_probability

    # test sampling
    from numpy.random import RandomState

    def assert_samples_ok(samples):
        """
        Test multinomial random variate generation.
        """

        from nose.tools import assert_almost_equal

        totals = numpy.sum(numpy.asarray(samples), 0, float)

        assert_almost_equal(totals[0] / len(samples), 0.25, places = 2)
        assert_almost_equal(totals[1] / len(samples), 0.75, places = 2)

    random = RandomState(42)

    yield assert_samples_ok, [m.random_variate(random) for i in xrange(4096)]
    yield assert_samples_ok, m.random_variates(32768, random)

def test_multinomial_estimator():
    """
    Test estimation of the multinomial distribution.
    """

    # generate some data
    import numpy

    vectors = numpy.empty((100, 2), numpy.uint)

    vectors[:75] = numpy.array([1, 0])
    vectors[75:] = numpy.array([0, 1])

    # verify basic estimator behavior
    from nose.tools                   import assert_almost_equal
    from cargo.statistics.multinomial import MultinomialEstimator

    estimator   = MultinomialEstimator(epsilon = 0.0)
    multinomial = estimator.estimate(vectors)

    assert_almost_equal(multinomial.beta[0], 0.75)
    assert_almost_equal(multinomial.beta[1], 0.25)

    # verify weighted estimator behavior
    weights     = numpy.array(([0.25] * 75) + ([0.75] * 25))
    multinomial = estimator.estimate(vectors, weights = weights)

    assert_almost_equal(multinomial.beta[0], 0.5)
    assert_almost_equal(multinomial.beta[1], 0.5)

