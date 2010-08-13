"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log             import get_logger
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

log = get_logger(__name__)

def smooth_multinomial_mixture(mixture, epsilon = 1e-3):
    """
    Apply a smoothing term to the multinomial mixture components.
    """

    log.info("heuristically smoothing a multinomial mixture")

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            beta                      = mixture.components[m, k].beta + epsilon
            beta                     /= numpy.sum(beta)
            mixture.components[m, k]  = Multinomial(beta)

class Multinomial(Distribution):
    """
    The multinomial distribution.

    Relevant types:
        - sample: D-shaped uint ndarray
        - sequence: ND-shaped uint ndarray
    """

    def __init__(self, beta, norm = 1):
        """
        Instantiate the distribution.

        @param beta: The distribution parameter vector.
        """

        # initialization
        self._beta     = numpy.asarray(beta)
        self._log_beta = numpy.nan_to_num(numpy.log(self._beta))
        self._norm     = norm

        # let's not let us be idiots
        self._beta.flags.writeable     = False
        self._log_beta.flags.writeable = False

    def random_variate(self, random = numpy.random):
        """
        Return a sample from this distribution.
        """

        return random.multinomial(self._norm, self._beta).astype(numpy.uint)

    def random_variates(self, size, random = numpy.random):
        """
        Return an array of samples from this distribution.
        """

        return random.multinomial(self._norm, self._beta, size).astype(numpy.uint)

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        from cargo.statistics._multinomial import multinomial_log_probability

        return multinomial_log_probability(self._log_beta, sample)

    def total_log_likelihood(self, samples):
        """
        Return the log likelihood of C{samples} under this distribution.
        """

        return self.log_likelihood(numpy.sum(samples, 0))

    @property
    def beta(self):
        """
        Return the multinomial parameter vector.
        """

        return self._beta

    @property
    def log_beta(self):
        """
        Return the multinomial log parameter vector.
        """

        return self._log_beta

class MultinomialEstimator(Estimator):
    """
    Estimate the parameters of a multinomial distribution.
    """

    def __init__(self, norm = 1):
        """
        Initialize.
        """

        self._norm = norm

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

        return Multinomial(mean, self._norm)

