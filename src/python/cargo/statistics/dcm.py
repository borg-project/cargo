"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from cargo.log             import get_logger
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )
from cargo.statistics._dcm import DirichletCompoundMultinomial

log = get_logger(__name__)

# FIXME should be some visitorish thing?
def smooth_dcm_mixture(mixture, epsilon = 1e-6):
    """
    Apply a smoothing term to the DCM mixture components.
    """

    # find the smallest non-zero dimension
    smallest = numpy.inf

    for components in mixture.components:
        for component in components:
            for v in component.alpha:
                if v < smallest and v > epsilon:
                    smallest = v

    if numpy.isinf(smallest):
        smallest = epsilon

    log.debug("smallest nonzero value is %f", smallest)

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            alpha                    = mixture.components[m, k].alpha
            mixture.components[m, k] = DirichletCompoundMultinomial(alpha + smallest * 1e-2)

class MinkaFixedPointEstimator(Estimator):
    """
    Estimate the parameters of a DCM distribution using Minka's fixed point iteration.

    Extended to allow sample weighting for expectation maximization in mixture models.
    """

    def __init__(self, norm = 1, threshold = 1e-6, cutoff = 1e5):
        """
        Initialize.
        """

        self._norm      = norm
        self._threshold = threshold
        self._cutoff    = int(cutoff)

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated maximum likelihood distribution.
        """

        # parameters
        if weights is None:
            weights = numpy.ones(samples.shape[0])
        else:
            weights = numpy.asarray(weights)

        # estimate
        from cargo.statistics._dcm import estimate_dcm_minka_fixed

        alpha = estimate_dcm_minka_fixed(samples, weights, self._threshold, self._cutoff)

        return DirichletCompoundMultinomial(alpha, self._norm)

class WallachRecurrenceEstimator(Estimator):
    """
    Estimate the parameters of a DCM distribution using Wallach's digamma
    recurrence iteration.

    Extended to allow sample weighting for expectation maximization in mixture
    models.
    """

    def __init__(self, norm = 1, threshold = 1e-5, cutoff = 1e3):
        """
        Initialize.
        """

        self._norm      = norm
        self._threshold = threshold
        self._cutoff    = int(cutoff)

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated maximum likelihood distribution.
        """

        # parameters
        samples = numpy.asarray(samples, numpy.uint)

        if weights is None:
            weights = numpy.ones(samples.shape[0])
        else:
            weights = numpy.asarray(weights)

        # counts are available; estimate
        from cargo.statistics._dcm import estimate_dcm_wallach_recurrence

        alpha = \
            estimate_dcm_wallach_recurrence(
                samples,
                weights,
                self._threshold,
                self._cutoff,
                )

        return DirichletCompoundMultinomial(alpha, self._norm)

# select the "best" estimator
DCM_Estimator = WallachRecurrenceEstimator

