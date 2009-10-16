"""
utexas/statistics/distribution.py

The distribution ABC.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import (
    ABCMeta,
    abstractmethod)
from itertools import count
from numpy import newaxis
from utexas.alog import DefaultLogger

log = DefaultLogger("utexas.statistics.distribution")

class Family(object):
    """
    The ABC for probability distributions.

    A distribution object is an instantiation of a distribution (eg N(1.0, 1.0));
    a distribution class corresponds to a family of distributions.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def random_variate(self):
        """
        Return a sample from this distribution.
        """

        pass

    @abstractmethod
    def random_variates(self, N, T):
        """
        Return an array of samples from this distribution.

        @param N: The L1 norm of the count vectors drawn.
        @param T: The number of count vectors to draw.
        """

        pass

    @abstractmethod
    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{counts} under this distribution.
        """

        pass

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of many samples from this distribution.
        """

        return sum(self.log_likelihood(s) for s in samples)

class Estimator(object):
    """
    Estimate a distribution from samples.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def estimate(self, samples):
        """
        Return the estimated distribution.
        """

        pass

