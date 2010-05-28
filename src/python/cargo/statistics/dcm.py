"""
cargo/statistics/dcm.py

The Dirichlet compound multinomial (DCM) distribution.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from functools import partial
from cargo.log import get_logger
from cargo.statistics._statistics import (
    dcm_log_probability,
    estimate_dcm_minka_fixed,
    estimate_dcm_wallach_recurrence,
    )
from cargo.statistics.distribution import (
#     Family,
    Estimator,
    )

log = get_logger(__name__)

def smooth_dcm_mixture(mixture):
    """
    Apply a smoothing term to the DCM mixture components.
    """

    # find the smallest non-zero dimension
    smallest = numpy.inf
    epsilon  = 1e-6

    for components in mixture.components:
        for component in components:
            for v in component.alpha:
                if v < smallest and v > epsilon:
                    smallest = v

    if numpy.isinf(smallest):
        smallest = epsilon

    log.debug("smallest nonzero value is %f", smallest)

    for m in xrange(mixture.ndomains):
        for k in xrange(mixture.ncomponents):
            alpha                    = mixture.components[m, k].alpha
            mixture.components[m, k] = DirichletCompoundMultinomial(alpha + smallest * 1e-2)

# class DirichletCompoundMultinomial(Family):
class DirichletCompoundMultinomial(object):
    """
    The Dirichlet Compound Multinomial (DCM) distribution.
    """

    def __init__(self, alpha, renorm = True):
        """
        Instantiate the distribution.

        @param alpha: The distribution parameter vector.
        @param renorm: Ensure that the parameter vector has some minimum L1 norm?
        """

        # initialization
        alpha = numpy.asarray(alpha)

        if renorm and numpy.sum(alpha) < 1e-2:
            self.__alpha = 1e-2 * (alpha / numpy.sum(alpha))
        else:
            self.__alpha = alpha

        self.sum_alpha = numpy.sum(self.__alpha)
#         self.log_likelihood = partial(dcm_log_probability, self.__sum_alpha, self.__alpha)

        # let's not let us be idiots
        self.__alpha.flags.writeable = False

    def random_variate(self, N):
        """
        Return a sample from this distribution.

        @param N: The L1 norm of the count vectors drawn.
        """

        beta = scipy.random.dirichlet(self.__alpha)

        if numpy.sum(beta[:-1]) > 1.0:
            beta /= numpy.sum(beta)

        return scipy.random.multinomial(N, beta)

    def random_variates(self, N, T):
        """
        Return an array of samples from this distribution.

        @param N: The L1 norm of the count vectors drawn.
        @param T: The number of count vectors to draw.
        """

        variates = numpy.empty((T,) + self.__alpha.shape)

        for t in xrange(T):
            variates[t] = self.random_variate(N)

        return variates

    def log_likelihood(self, counts):
        """
        Return the log likelihood of C{counts} under this distribution.
        """

        return dcm_log_probability(self.sum_alpha, self.__alpha, counts)

    def __get_shape(self):
        """
        Return the tuple of the dimensionalities of this distribution.
        """

        return self.__alpha.shape

    def __get_alpha(self):
        """
        Return the Dirichlet parameter vector.
        """

        return self.__alpha

    def __get_mean(self):
        """
        Return the mean of the distribution.
        """

        return self.__alpha / numpy.sum(self.__alpha)

    def __get_burstiness(self):
        """
        Return the "burstiness" of the distribution.
        """

        return numpy.sum(self.__alpha)

    # properties
    shape      = property(__get_shape)
    alpha      = property(__get_alpha)
    mean       = property(__get_mean)
    burstiness = property(__get_burstiness)

class MinkaFixedPointEstimator(Estimator):
    """
    Estimate the parameters of a DCM distribution using Minka's fixed point iteration.

    Extended to allow sample weighting for expectation maximization in mixture models.
    """

    def estimate(self, counts, weights = None, threshold = 1e-6, cutoff = 1e5):
        """
        Return the estimated maximum likelihood distribution.
        """

        # sanity
        assert cutoff >= 1

        # parameters
        if weights is None:
            weights = numpy.ones(counts.shape[0])
        else:
            weights = numpy.asarray(weights, dtype = numpy.float)

        counts = numpy.asarray(counts, dtype = numpy.uint32)

        # estimate
        alpha = estimate_dcm_minka_fixed(counts, weights, threshold, int(cutoff))

        return DirichletCompoundMultinomial(alpha)

class WallachRecurrenceEstimator(Estimator):
    """
    Estimate the parameters of a DCM distribution using Wallach's digamma
    recurrence iteration.

    Extended to allow sample weighting for expectation maximization in mixture
    models.
    """

    def estimate(self, counts, weights = None, threshold = 1e-5, cutoff = 1e3):
        """
        Return the estimated maximum likelihood distribution.
        """

        # sanity
        assert cutoff >= 1

        # parameters
        if weights is None:
            weights = numpy.ones(counts.shape[0])
        else:
            weights = numpy.asarray(weights, dtype = numpy.float)

        counts = numpy.asarray(counts, dtype = numpy.uint32)

        # FIXME hackishly handle the zero-counts case
        nonzero = numpy.sum(counts, 1) > 0

        if counts[nonzero].size == 0:
            # no counts; be uninformative
            return DirichletCompoundMultinomial(numpy.ones(counts.shape[1]) * 1e6)
        else:
            # counts are available; estimate
            alpha = \
                estimate_dcm_wallach_recurrence(
                    counts[nonzero],
                    weights[nonzero],
                    threshold,
                    int(cutoff),
                    )

            return DirichletCompoundMultinomial(alpha)

# select the "best" estimator
DCM_Estimator = WallachRecurrenceEstimator

