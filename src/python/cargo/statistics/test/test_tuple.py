"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy.random           import RandomState
from nose.tools             import assert_almost_equal
from cargo.statistics.tuple import TupleDistribution

def test_tuple_rv():
    """
    Test random variate generation under the tuple distribution.
    """

    from cargo.statistics.discrete import Discrete

    def assert_samples_ok(samples):
        """
        Assert that the generated samples look reasonable.
        """


    r = RandomState(42)
    d = TupleDistribution((Discrete([0.1, 0.9]), Discrete([0.9, 0.1])))

    assert_samples_ok([distribution.random_variate(random) for _ in xrange(4096)])
    assert_samples_ok(distribution.random_variates(4096, random))

    assert_almost_equal(float(sum(s[0] for s in samples)) / len(samples), 0.9, places = 2)
    assert_almost_equal(float(sum(s[1] for s in samples)) / len(samples), 0.1, places = 2)

def test_tuple_ll():
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

def test_tuple_estimator():
    """
    Test estimation of the tuple distribution.
    """

    import numpy

    from nose.tools                import assert_almost_equal
    from cargo.statistics.tuple    import TupleEstimator
    from cargo.statistics.discrete import DiscreteEstimator

    estimator = TupleEstimator((DiscreteEstimator(2), DiscreteEstimator(2)))
    samples   = [(0, 1)] * 2500 + [(1, 0)] * 7500
    estimated = estimator.estimate(samples)

    assert_almost_equal(estimated.inner[0].beta[0], 0.25)
    assert_almost_equal(estimated.inner[0].beta[1], 0.75)
    assert_almost_equal(estimated.inner[1].beta[0], 0.75)
    assert_almost_equal(estimated.inner[1].beta[1], 0.25)

    estimated = estimator.estimate([(0, 1), (1, 0)], weights = numpy.array([0.25, 0.75]))

    assert_almost_equal(estimated.inner[0].beta[0], 0.25)
    assert_almost_equal(estimated.inner[0].beta[1], 0.75)
    assert_almost_equal(estimated.inner[1].beta[0], 0.75)
    assert_almost_equal(estimated.inner[1].beta[1], 0.25)

