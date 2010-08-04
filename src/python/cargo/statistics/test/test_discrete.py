"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_discrete_distribution():
    """
    Test operations on the discrete distribution.
    """

    import numpy

    from numpy.random              import RandomState
    from cargo.statistics.discrete import Discrete

    random       = RandomState(42)
    distribution = Discrete([0.25, 0.75])

    def test_random_variate():
        """
        Test random variate generation under the discrete distribution.
        """

        from nose.tools import assert_almost_equal

        samples = distribution.random_variates(1024, random)
        counts  = numpy.zeros(2)

        for k in samples:
            counts[k] += 1

        assert_almost_equal(counts[0] / 1024.0, 0.25, places = 2)
        assert_almost_equal(counts[1] / 1024.0, 0.75, places = 2)

    def test_log_likelihood():
        """
        Test log probability computation under the discrete distribution.
        """

        from nose.tools import assert_almost_equal

        assert_almost_equal(distribution.log_likelihood(0), numpy.log(0.25))
        assert_almost_equal(distribution.log_likelihood(1), numpy.log(0.75))

def test_discrete_estimator():
    """
    Test estimation of the discrete distribution.
    """

    import numpy

    from nose.tools                import assert_almost_equal
    from cargo.statistics.discrete import DiscreteEstimator

    samples   = numpy.array([0] * 25 + [1] * 75)
    estimator = DiscreteEstimator(2)
    estimated = estimator.estimate(samples)

    assert_almost_equal(estimated.beta[0], 0.25)
    assert_almost_equal(estimated.beta[1], 0.75)

