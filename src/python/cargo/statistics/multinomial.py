"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy                         import newaxis
from cargo.log                     import get_logger
from cargo.statistics._multinomial import multinomial_log_probability
from cargo.statistics.distribution import Estimator

log = get_logger(__name__)

def smooth_multinomial_mixture(mixture):
    """
    Apply a smoothing term to the multinomial mixture components.
    """

    log.info("heuristically smoothing a multinomial mixture")

    epsilon = 1e-3

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            beta                      = mixture.components[m, k].beta + epsilon
            beta                     /= numpy.sum(beta)
            mixture.components[m, k]  = Multinomial(beta)

class Multinomial(object):
    """
    The multinomial distribution.
    """

    def __init__(self, beta, norm = 1):
        """
        Instantiate the distribution.

        @param beta: The distribution parameter vector.
        """

        # initialization
        self.__beta      = beta = numpy.asarray(beta)
        self.__beta     /= numpy.sum(beta)
        self.__log_beta  = numpy.nan_to_num(numpy.log(self.__beta))
        self.__norm      = norm

        # let's not let us be idiots
        self.__beta.flags.writeable     = False
        self.__log_beta.flags.writeable = False

    def random_variate(self, N = None, random = numpy.random):
        """
        Return a sample from this distribution.

        @param N: The L1 norm of the count vectors drawn.
        """

        if N is None:
            N = self.__norm

        return random.multinomial(N, self.__beta)

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

    def __get_log_beta(self):
        """
        Return the multinomial log parameter vector.
        """

        return self.__log_beta

    # properties
    shape    = property(__get_shape)
    beta     = property(__get_beta)
    log_beta = property(__get_log_beta)
    mean     = beta

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

        weights   = numpy.ones(counts.shape[0]) if weights is None else weights
        weighted  = counts * weights[:, newaxis]
        mean      = numpy.sum(weighted, 0)
        mean     /= numpy.sum(mean)

        return Multinomial(mean)

    def random_estimate(self, D):
        """
        Return a randomly-initialized distribution.
        """

        beta  = numpy.random.random(D)
        beta /= numpy.sum(beta)

        return Multinomial(beta)

