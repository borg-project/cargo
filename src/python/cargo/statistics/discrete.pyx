# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from collections           import Sequence
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

cimport numpy
cimport cython

from cargo.gsl.sf cimport log

class Discrete(Distribution):
    """
    The discrete distribution.

    Relevant types:
        - sample: int (outcome index)
        - sequence: N-shaped uint ndarray
    """

    def __init__(self, beta):
        """
        Instantiate the distribution.

        @param beta: The distribution parameter vector.
        """

        self._beta = numpy.asarray(beta)

    def random_variate(self, random = numpy.random):
        """
        Return a sample from this distribution.
        """

        # mise en place
        cdef numpy.ndarray[double, ndim = 1] beta = self._beta

        # sample from this distribution
        cdef size_t i
        cdef double r = random.rand()

        for i in xrange(beta.shape[0] - 1):
            if r < beta[i]:
                return i

        return beta.shape[0] - 1

    def log_likelihood(self, int sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        return log(self._beta[sample])

    @cython.boundscheck(False)
    def add_log_likelihoods(self, samples, to):
        """
        Add the log likelihoods of C{samples} under this distribution.
        """

        # mise en place
        cdef numpy.ndarray[double]       beta_D    = self._beta
        cdef numpy.ndarray[numpy.uint_t] samples_N = numpy.asarray(samples, numpy.uint)
        cdef numpy.ndarray[double]       to_N      = to

        assert samples_N.shape[0] == to_N.shape[0]

        # calculate
        cdef size_t D = beta_D.shape[0]
        cdef size_t n
        cdef size_t s

        for n in xrange(samples_N.shape[0]):
            s = samples_N[n]

            if s >= D:
                raise ValueError("discrete-draw index out of bounds")
            else:
                to[n] += beta_D[s]

    @property
    def beta(self):
        """
        Return the underlying distribution parameter vector.
        """

        return self._beta

class DiscreteEstimator(Estimator):
    """
    A maximum-likelihood estimator of discrete distributions.
    """

    def __init__(self, D, epsilon = 1e-3):
        """
        Initialize.

        @param D: The dimensionality of beta.
        """

        self._D       = D
        self._epsilon = epsilon

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        # samples array
        cdef numpy.ndarray[numpy.uint_t] samples_N = numpy.asarray(samples, numpy.uint)

        # weights array
        cdef numpy.ndarray[double] weights_N

        if weights is None:
            weights_N = numpy.ones(len(samples))
        else:
            weights_N = numpy.asarray(weights)

        assert samples_N.shape[0] == weights_N.shape[0]

        # build the estimate
        cdef size_t                D      = self._D
        cdef numpy.ndarray[double] beta_D = numpy.zeros(D)
        cdef numpy.uint_t          d
        cdef size_t                n

        for n in xrange(samples_N.shape[0]):
            d = samples_N[n]

            if d < D:
                beta_D[d] += weights_N[n]
            else:
                raise ValueError("invalid discrete sample index")

        if self._epsilon is not None:
            beta_D += self._epsilon

        beta_D /= numpy.sum(beta_D)

        return Discrete(beta_D)

class ObjectDiscrete(Distribution):
    """
    The discrete distribution over an arbitrary domain.

    Relevant types:
        - sample: object
        - sequence: list
    """

    def __init__(self, beta, domain):
        """
        Instantiate the distribution.

        @param beta:   The distribution parameter vector.
        @param domain: The sample domain.
        """

        self._discrete = Discrete(beta)
        self._domain   = domain

    def random_variate(self, random = numpy.random):
        """
        Return a sample from this distribution.
        """

        return self._domain[self._discrete.random_variate(random)]

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        return self._discrete.log_likelihood(self._domain.index(sample))

    @property
    def beta(self):
        """
        Return the underlying distribution parameter vector.
        """

        return self._discrete.beta

    @property
    def domain(self):
        """
        Return the associated domain.
        """

        return self._domain

class ObjectDiscreteEstimator(Estimator):
    """
    A maximum-likelihood estimator of discrete distributions.
    """

    def __init__(self, domain, epsilon = 1e-3):
        """
        Initialize.
        """

        self._domain    = domain
        self._estimator = DiscreteEstimator(len(domain), epsilon = epsilon)

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        indices = numpy.array(map(self._domain.index, samples), numpy.uint)

        return self._estimator.estimate(indices, random, weights)

