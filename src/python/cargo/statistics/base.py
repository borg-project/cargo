"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import abstractmethod
from cargo.sugar import ABC

class Distribution(ABC):
    """
    Interface to a probability distribution.
    """

    @abstractmethod
    def random_variate(self, random = numpy.random):
        """
        Return a single sample from this distribution.
        """

    def random_variates(self, size, random = numpy.random):
        """
        Return a sequence of independent samples from this distribution.

        @return: A collection of values that supports the Sequence interface.
        """

        return [self.random_variate(random) for i in xrange(size)]

    @abstractmethod
    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.

        @param samples: An arbitrary sample value.
        """

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of C{samples} under this distribution.

        @param samples: A value supporting the Sequence interface.
        """

        return sum(self.log_likelihood(s) for s in samples)

class Estimator(ABC):
    """
    Interface to a maximum-likelihood estimator of distributions.
    """

    @abstractmethod
    def estimate(self, samples, random = numpy.random):
        """
        Return the estimated distribution.
        """

