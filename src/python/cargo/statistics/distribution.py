"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.sugar import ABC

class Estimator(ABC):
    """
    Estimate a distribution from samples.
    """

    @abstractmethod
    def estimate(self, samples):
        """
        Return the estimated distribution.
        """

        pass

class TupleDistribution(object):
    """
    Generate samples from a tuple of independent distributions.
    """

    def __init__(self, distributions):
        """
        Initialize.
        """

        self._distributions = distributions

    def random_variate(self):
        """
        Return a sample from this distribution.
        """

        return scipy.random.multinomial(N, self.__beta)

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        pass

    @property
    def distributions(self):
        """
        Get this distribution's component distributions.
        """

        return self._distributions

class ConstantDistribution(object):
    """
    The trivial fixed constant distribution.
    """

    def __init__(self, constant):
        """
        Initialize.
        """

        self._constant = constant

    def random_variate(self):
        """
        Return the constant.
        """

        return self._constant

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        if sample == self._constant:
            return 0.0
        else:
            import numpy

            return numpy.finfo(float).min

