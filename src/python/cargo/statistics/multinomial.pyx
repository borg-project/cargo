"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

cimport numpy

from libc.float   cimport DBL_MAX
from cargo.gsl.sf cimport (
    log,
    ln_gamma,
    )

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
        self._norm     = norm

        # let's not let ourselves be idiots
        self._beta.flags.writeable     = False

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
        Return the log probability of C{sample} under this distribution.
        """

        # mise en place
        cdef numpy.ndarray[double       , ndim = 1] beta_D   = self._beta
        cdef numpy.ndarray[unsigned long, ndim = 1] counts_D = sample

        assert counts_D.shape[0] == beta_D.shape[0]

        # calculate
        cdef double        lp = 0.0
        cdef unsigned long n  = 0

        for d in xrange(counts_D.shape[0]):
            n += counts_D[d]

            if beta_D[d] > 0.0:
                lp += log(beta_D[d]) * counts_D[d] - ln_gamma(counts_D[d] + 1)
            else:
                lp += -DBL_MAX * counts_D[d] - ln_gamma(counts_D[d] + 1)

        return lp + ln_gamma(n + 1)

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

        return numpy.nan_to_num(numpy.log(self._beta))

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

