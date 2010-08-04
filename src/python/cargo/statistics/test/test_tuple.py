"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_tuple_distribution():
    """
    Test operations on the tuple distribution.
    """

    import numpy

    from nose.tools                import assert_almost_equal
    from numpy.random              import RandomState
    from cargo.statistics.tuple    import TupleDistribution
    from cargo.statistics.discrete import Discrete

    random       = RandomState(42)
    distribution = TupleDistribution((Discrete([0.1, 0.9]), Discrete([0.9, 0.1])))

    def assert_samples_ok(samples):
        """
        Assert that the generated samples look reasonable.
        """

        assert_almost_equal(float(sum(s[0] for s in samples)) / len(samples), 0.9, places = 2)
        assert_almost_equal(float(sum(s[1] for s in samples)) / len(samples), 0.1, places = 2)

    def test_random_variate():
        """
        Test random variate generation under the tuple distribution.
        """

        assert_samples_ok([distribution.random_variate(random) for _ in xrange(4096)])
        assert_samples_ok(distribution.random_variates(4096, random))

    yield test_random_variate

    def test_log_likelihood():
        """
        Test log likelihood computation under the tuple distribution.
        """

        assert_almost_equal(
            distribution.log_likelihood((1, 1)),
            numpy.log(0.1 * 0.9),
            )
        assert_almost_equal(
            distribution.total_log_likelihood([(1, 1), (0, 0)]),
            numpy.log((0.1 * 0.9)**2),
            )

    yield test_log_likelihood

