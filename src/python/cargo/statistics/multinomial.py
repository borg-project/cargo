"""
utexas/statistics/multinomial.py

The multinomial distribution.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from numpy import newaxis
from scipy.special.basic import gammaln
from utexas.alog import DefaultLogger
from utexas.statistics._statistics import multinomial_log_probability
from utexas.statistics.distribution import (
    Family,
    Estimator)

log = DefaultLogger("utexas.statistics.multinomial")

class Multinomial(Family):
    """
    The multinomial distribution.
    """

    def __init__(self, beta):
        """
        Instantiate the distribution.

        @param beta: The distribution parameter vector.
        """

        self.__beta = beta = numpy.asarray(beta)
        self.__beta /= numpy.sum(beta)
        self.__log_beta = numpy.nan_to_num(numpy.log(self.__beta))

    def random_variate(self, N):
        """
        Return a sample from this distribution.

        @param N: The L1 norm of the count vectors drawn.
        """

        return scipy.random.multinomial(N, self.__beta)

    def random_variates(self, N, T):
        """
        Return an array of samples from this distribution.

        @param N: The L1 norm of the count vectors drawn.
        @param T: The number of count vectors to draw.
        """

        return numpy.random.multinomial(N, self.__beta, T)

    def log_likelihood(self, counts):
        """
        Return the log likelihood of C{counts} under this distribution.

        >>> m = Multinomial([0.5, 0.5])
        >>> lp = m.log_likelihood(numpy.array([1, 0], dtype = numpy.uint))
        >>> numpy.exp(lp)
        0.5
        >>> lp = m.log_likelihood(numpy.array([0, 2], dtype = numpy.uint))
        >>> numpy.exp(lp)
        0.25
        """

        return multinomial_log_probability(self.__log_beta, counts)

    def __get_shape(self):
        """
        Return the tuple of the dimensionalities of this distribution.
        """

        return self.__beta.shape

    def __get_beta(self):
        """
        Return the multinomial parameter vector.
        """

        return self.__beta

    # properties
    shape = property(__get_shape)
    beta = property(__get_beta)
    mean = beta

class MultinomialEstimator(Estimator):
    """
    Estimate the parameters of a multinomial distribution.

    Extended to allow sample weighting for expectation maximization in mixture models.
    """

    def __init__(self):
        """
        Initialize.
        """

        pass

    def estimate(self, counts, weights = None, verbose = False):
        """
        Return the estimated maximum likelihood distribution.
        """

        weights = numpy.ones(counts.shape[0]) if weights is None else weights
        weighted = counts * weights[:, newaxis]
        mean = numpy.sum(weighted, 0)
        mean /= numpy.sum(mean)

        return Multinomial(mean)

