"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log             import get_logger
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )
from cargo.statistics._multinomial import Multinomial

log = get_logger(__name__)

class MultinomialEstimator(Estimator):
    """
    Estimate the parameters of a multinomial distribution.
    """

    def __init__(self, norm = 1, epsilon = 1e-3):
        """
        Initialize.
        """

        self._norm    = norm
        self._epsilon = epsilon

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated maximum likelihood distribution.
        """

        # parameters
        samples = numpy.asarray(samples)

        if weights is None:
            weights = numpy.ones(samples.shape[0])
        else:
            weights = numpy.asarray(weights)

        # estimate
        mean  = numpy.sum(samples * weights[:, None], 0)
        mean /= numpy.sum(mean)

        # heuristic smoothing, if requested
        if self._epsilon is not None:
            mean += self._epsilon
            mean /= numpy.sum(mean)

        return Multinomial(mean, self._norm)

