# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

cimport numpy
cimport cython

class Gaussian(Distribution):
    """
    The univariate normal distribution.

    Relevant types:
    - sample   : double
    - sequence : N-shaped double ndarray
    """

    def __init__(self, mu, sigma, norm = 1):
        """
        Instantiate the distribution.

        @param beta: The distribution parameter vector.
        """

        self._mu    = mu
        self._sigma = sigma

    def random_variate(self, random = numpy.random):
        """
        Return a sample from this distribution.
        """

        return random.normal(self._mu, self._sigma)

    def random_variates(self, size, random = numpy.random):
        """
        Return an array of samples from this distribution.
        """

        return random.normal(self._mu, self._sigma, size)

    def log_likelihood(self, double sample):
        """
        Return the log probability of C{sample} under this distribution.
        """

        return numpy.log(scipy.stats.norm.pdf(sample, loc = self._mu, scale = self._sigma))

    def add_log_likelihoods(self, samples, to):
        """
        Add the log likelihoods of C{samples} under this distribution.
        """

        p = scipy.stats.norm.pdf(samples, loc = self._mu, scale = self._sigma)

        numpy.log(p, p)

        to += p

    def total_log_likelihood(self, samples):
        """
        Return the log likelihood of C{samples} under this distribution.
        """

        p = scipy.stats.norm.pdf(samples, loc = self._mu, scale = self._sigma)

        numpy.log(p, p)

        return numpy.sum(p)

    @property
    def mu(self):
        """
        Return the distribution mean.
        """

        return self._mu

    @property
    def sigma(self):
        """
        Return the distribution variance.
        """

        return self._sigma

class GaussianEstimator(Estimator):
    """
    A maximum-likelihood estimator of Gaussian distributions.
    """

    def __init__(self):
        """
        Initialize.
        """

    @cython.boundscheck(False)
    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        # input arrays
        cdef numpy.ndarray[double] weights_N
        cdef numpy.ndarray[double] samples_N = numpy.asarray(samples)

        if weights is None:
            weights_N = numpy.ones(len(samples))
        else:
            weights_N = numpy.asarray(weights)

        assert samples_N.shape[0] == weights_N.shape[0]

        # estimate the mean
        cdef double mu     = 0.0
        cdef double weight = 0.0
        cdef size_t n

        for n in xrange(samples_N.shape[0]):
            mu     += weights_N[n] * samples_N[n]
            weight += weights_N[n]

        mu /= weight

        # estimate the variance
        cdef double sigma  = 0.0

        for n in xrange(samples_N.shape[0]):
            sigma += weights_N[n] * (samples_N[n] - mu)**2.0

        sigma /= weight

        return Gaussian(mu, sigma)

