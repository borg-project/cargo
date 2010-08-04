"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from collections           import Sequence
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

class Discrete(Distribution):
    """
    The discrete distribution.

    Relevant types:
        - samples: int (outcome index)
        - sample sequences: N-shaped uint ndarray
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

        ((k,),) = numpy.nonzero(random.multinomial(1, self._beta))

        return k

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        return numpy.log(self._beta[sample])

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

    def estimate(self, samples, random = numpy.random):
        """
        Return the estimated distribution.
        """

        counts = numpy.zeros(self._D)

        for k in samples:
            counts[k] += 1

        return Discrete(counts / len(samples))

class ObjectDiscrete(Distribution):
    """
    The discrete distribution over an arbitrary domain.

    Relevant types:
        - samples: object
        - sample sequences: list
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

    def estimate(self, samples, random = numpy.random):
        """
        Return the estimated distribution.
        """

        indices = \
            numpy.fromiter(
                (self._domain.index(s) for s in samples),
                numpy.uint,
                len(samples),
                )

        return self._estimator.estimate(indices)

