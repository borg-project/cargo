"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from cargo.log import get_logger

log = get_logger(__name__)

def test_dcm_random_variate():
    """
    Test operations on the DCM distribution.
    """

    from numpy.random         import RandomState
    from cargo.statistics.dcm import DirichletCompoundMultinomial

    random = RandomState(6995749)
    dcm    = DirichletCompoundMultinomial([0.1, 1.0], 8)

    def assert_samples_ok(samples):
        """
        Assert that the sample statistics appear reasonable.
        """

        from nose.tools import assert_almost_equal

        mean = numpy.sum(samples, 0, float) / len(samples)

        assert_almost_equal(mean[0], 0.724, places = 2)
        assert_almost_equal(mean[1], 7.275, places = 2)

    yield assert_samples_ok, numpy.array([dcm.random_variate(random) for _ in xrange(65536)])
    yield assert_samples_ok, dcm.random_variates(65536, random)

def verified_dcm_log_likelihood(alpha, bins):
    """
    Return the log likelihood of C{bins} under the DCM.
    """

    from cargo.statistics.functions import ln_poch

    u_lnp = numpy.frompyfunc(ln_poch, 2, 1)
    psigm = numpy.sum(u_lnp(alpha, bins))
    clens = numpy.sum(bins)
    alsum = numpy.sum(alpha)
    nsigm = u_lnp(alsum, clens)

    return numpy.sum(psigm - nsigm)

def assert_log_likelihood_ok(alpha, sample, ll):
    """
    Assert that the specified log likelihood is correct.
    """

    from nose.tools import assert_almost_equal

    assert_almost_equal(
        ll,
        verified_dcm_log_likelihood(alpha, sample),
        )

def test_dcm_log_likelihood():
    """
    Test log-likelihood computation under the DCM.
    """

    def test_inner(alpha, sample):
        """
        Test log-likelihood computation under the DCM.
        """

        from cargo.statistics.dcm import DirichletCompoundMultinomial

        sample = numpy.asarray(sample, numpy.uint)
        dcm    = DirichletCompoundMultinomial(alpha, numpy.sum(sample))

        assert_log_likelihood_ok(alpha, sample, dcm.log_likelihood(sample))

    yield test_inner, [0.1, 1.0], [1, 1]
    yield test_inner, [0.1, 1.0], [2, 3]
    yield test_inner, [0.1, 1.0], [8, 0]

def verified_dcm_estimate(counts, weights, threshold, cutoff):
    """
    Return an estimated maximum likelihood distribution.
    """

    def alpha_new(alpha, counts, weights, total_weight):
        """
        Compute the next value in the fixed-point iteration.
        """

        from numpy               import newaxis
        from scipy.special.basic import psi

        N = counts.shape[0]
        clens = numpy.sum(counts, 1)
        alsum = numpy.sum(alpha)
        numer = numpy.sum(psi(counts + alpha) * weights[:, newaxis], 0) - total_weight * psi(alpha)
        denom = numpy.sum(psi(clens + alsum) * weights, 0) - total_weight * psi(alsum)

        return alpha * numer / denom

    # massage the inputs
    weights = numpy.asarray(weights, dtype = numpy.float)
    counts  = numpy.asarray(counts, dtype  = numpy.uint)
    alpha   = numpy.ones(counts.shape[1])

    # set up the iteration and go
    from itertools import count

    total_weight = numpy.sum(weights)

    for i in count(1):
        old = alpha
        alpha = alpha_new(old, counts, weights, total_weight)
        difference = numpy.sum(numpy.abs(old - alpha))
        alpha[alpha < 1e-16] = 1e-16

        if difference < threshold or (cutoff is not None and i >= cutoff):
            return alpha

def assert_estimator_ok(estimator, counts, weights):
    """
    Assert that the DCM estimator provides the verified result.
    """

    from nose.tools   import assert_true
    from numpy.random import RandomState

    counts         = numpy.asarray(counts, numpy.uint)
    verified_alpha = verified_dcm_estimate(counts, weights, 1e-6, 1e4)
    estimated_dcm  = estimator.estimate(counts, RandomState(1), weights)

    assert_true(numpy.allclose(estimated_dcm.alpha, verified_alpha))

def test_minka_fp_simple():
    """
    Test the Minka fixed-point estimator.
    """

    from cargo.statistics.dcm import MinkaFixedPointEstimator

    estimator = MinkaFixedPointEstimator(threshold = 1e-6, cutoff = 1e4)

    yield assert_estimator_ok, estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2)
    yield assert_estimator_ok, estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2) / 2.0
    yield assert_estimator_ok, estimator, [[0, 3], [3, 0], [9, 2]], [0.3, 0.7, 0.5]

def test_wallach_recurrence_simple():
    """
    Test the Wallach digamma-recurrence estimator.
    """

    from cargo.statistics.dcm import WallachRecurrenceEstimator

    estimator = WallachRecurrenceEstimator(threshold = 1e-6, cutoff = 1e4)

    yield assert_estimator_ok, estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2)
    yield assert_estimator_ok, estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2) / 2.0
    yield assert_estimator_ok, estimator, [[0, 3], [3, 0], [9, 2]], [0.3, 0.7, 0.5]

