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
        - sample sequences: N-dimensional ndarray
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

    def estimate(self, samples):
        """
        Return the estimated distribution.
        """

        counts = numpy.zeros(self._D)

        for k in samples:
            counts[k] += 1

        return Discrete(counts / len(samples))

