"""
utexas/statistics/test/test_dcm.py

Tests of the DCM distribution implementation.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from itertools                      import count
from numpy                          import newaxis
from scipy.special.basic            import psi
from cargo.log                      import get_logger
from cargo.statistics.dcm           import (
    MinkaFixedPointEstimator,
    WallachRecurrenceEstimator,
    DirichletCompoundMultinomial,
    )
from cargo.statistics.distribution  import Estimator

log = get_logger("utexas.statistics.test.test_dcm")

class VerifiedDCM(object):
    """
    Reasonably-tested but slow implementation of the DCM distribution.
    """

    def __init__(self, alpha):
        """
        Instantiate the distribution.
        """

        self.alpha = numpy.asarray(alpha)

    def random_variate(self, N):
        """
        Return a sample from this distribution.
        """

        beta = scipy.random.dirichlet(self.alpha)

        if numpy.sum(beta[:-1]) > 1.0:
            beta /= numpy.sum(beta)

        return scipy.random.multinomial(N, beta)

    def random_variates(self, N, T):
        """
        Return an array of samples from this distribution.
        """

        variates = numpy.empty((T,) + self.__alpha.shape)

        for t in xrange(T):
            variates[t] = self.random_variate(N)

        return variates

    def log_likelihood(self, bins):
        """
        Return the log likelihood of C{bins} under this distribution.
        """

        psigm = numpy.sum(pochhammer_ln(self.__alpha, bins))
        clens = numpy.sum(bins)
        alsum = numpy.sum(self.__alpha)
        nsigm = pochhammer_ln(alsum, clens)

        return numpy.sum(psigm - nsigm)

class VerifiedMFP(Estimator):
    """
    Reasonably-tested but slow implementation of the Minka fixed-point estimator.
    """

    def __alpha_new(self, alpha, counts, weights, total_weight):
        """
        Compute the next value in the fixed-point iteration.
        """

        # vectorized
        N = counts.shape[0]
        clens = numpy.sum(counts, 1)
        alsum = numpy.sum(alpha)
        numer = numpy.sum(psi(counts + alpha) * weights[:, newaxis], 0) - total_weight * psi(alpha)
        denom = numpy.sum(psi(clens + alsum) * weights, 0) - total_weight * psi(alsum)

        return alpha * numer / denom

    def estimate(self, counts, weights = None, threshold = None, cutoff = None):
        """
        Return the estimated maximum likelihood distribution.
        """

        # sanity
        assert cutoff >= 1

        # parameter normalization
        if weights is None:
            weights = numpy.ones(counts.shape[1])
        else:
            weights = numpy.asarray(weights, dtype = numpy.float)

        counts = numpy.asarray(counts, dtype = numpy.uint)
        alpha = numpy.ones(counts.shape[1])

        # set up the iteration and go
        total_weight = numpy.sum(weights)

        for i in count(1):
            old = alpha
            alpha = self.__alpha_new(old, counts, weights, total_weight)
            difference = numpy.sum(numpy.abs(old - alpha))
            alpha[alpha < 1e-16] = 1e-16

            if difference < threshold or (cutoff is not None and i >= cutoff):
                return DirichletCompoundMultinomial(alpha)

def assert_good_estimator(estimator, counts, weights):
    """
    Assert that the DCM estimator provides the verified result.
    """

    # set up estimators
    threshold = 1e-6
    cutoff = 1e4
    verified = VerifiedMFP()

    # generate estimates
    verified_dcm = verified.estimate(counts, weights = weights, threshold = threshold, cutoff = cutoff)
    estimated_dcm = estimator.estimate(counts, weights = weights, threshold = threshold, cutoff = cutoff)

    # compare estimates
    assert numpy.allclose(verified_dcm.alpha, estimated_dcm.alpha)

def test_minka_fp_simple():
    """
    Simple tests of the Minka fixed-point estimator.
    """

    estimator = MinkaFixedPointEstimator()

    assert_good_estimator(estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2))
    assert_good_estimator(estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2) / 2.0)
    assert_good_estimator(estimator, [[0, 3], [3, 0], [9, 2]], [0.3, 0.7, 0.5])

def test_wallach_recurrence_simple():
    """
    Simple tests of the Wallach digamma recurrence estimator.
    """

    estimator = WallachRecurrenceEstimator()

    assert_good_estimator(estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2))
    assert_good_estimator(estimator, numpy.arange(8).reshape((2, 4)), numpy.ones(2) / 2.0)
    assert_good_estimator(estimator, [[0, 3], [3, 0], [9, 2]], [0.3, 0.7, 0.5])

