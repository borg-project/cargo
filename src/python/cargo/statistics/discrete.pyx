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

    def __init__(self, D):
        """
        Initialize.

        @param D: The dimensionality of beta.
        """

        self._D = D

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        from itertools import izip

        if weights is None:
            weights = numpy.ones(len(samples))

        counts = numpy.zeros(self._D)

        for (k, w) in izip(samples, weights):
            counts[k] += w

        return Discrete(counts / numpy.sum(weights))

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

    def __init__(self, domain):
        """
        Initialize.
        """

        self._domain    = domain
        self._estimator = DiscreteEstimator(len(domain))

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        indices = numpy.array(map(self._domain.index, samples), numpy.uint)

        return self._estimator.estimate(indices, random, weights)

