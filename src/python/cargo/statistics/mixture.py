"""
cargo/statistics/mixture.py

Finite mixture distributions.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy import newaxis
from cargo.log import get_logger

log = get_logger(__name__)

class FiniteMixture(object):
    """
    An arbitrary finite [linked] mixture distribution.
    """

    def __init__(self, pi, components):
        """
        Initialize.
        """

        # basic members
        self.__pi_K = numpy.asarray(pi)

        # components array
        components_MK = numpy.asarray(components)

        if components_MK.ndim == 1:
            self.__components_MK = components_MK = components_MK[newaxis, :]
        elif components_MK.ndim == 2:
            self.__components_MK = components_MK
        else:
            raise ArgumentError("components list must be one- or two-dimensional")

        # sanity
        (M, K) = components_MK.shape

        assert M > 0
        assert K > 0
        assert self.__pi_K.shape == (K,)
        assert numpy.allclose(numpy.sum(self.__pi_K), 1.0)

        self.__shapes = [self.__components_MK[m, -1].shape for m in xrange(M)]

        for m in xrange(M):
            for k in xrange(K - 1):
                assert self.__components_MK[m, k].shape == self.__shapes[m]

    def random_variate(self, *args, **kwargs):
        """
        Make multiple draws from this mixture distribution.
        """

        (M, K) = self.__components_MK.shape
        ((k,),) = numpy.nonzero(numpy.random.multinomial(1, self.__pi_K))

        return [self.__components_MK[m, k].random_variate(*args, **kwargs) for m in xrange(M)]

    def random_variates(self, T, *args, **kwargs):
        """
        Make multiple draws from this mixture distribution.
        """

        variates = [numpy.empty((T,) + s) for s in self.__shapes]

        for t in xrange(T):
            draw = self.random_variate(*args, **kwargs)

            for m in xrange(self.ndomains):
                variates[m][t] = draw[m]

        return variates

    def log_likelihood(self, samples):
        """
        Return the log likelihood of C{samples} under this distribution.
        """

         # FIXME surely we can be more numerically clever here

        # parameters and sanity
        (M, K) = self.__components_MK.shape

        assert len(samples) == M

        # draw the sample(s)
        total = 0.0

        raise NotImplementedError() # FIXME I don't think that the expression below is correct

        for m in xrange(M):
            for k in xrange(K):
                total += self.__pi_K[k] * numpy.exp(self.__components_MK[m, k].log_likelihood(samples[m]))

        return numpy.log(total)

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of many C{samples} lists under this distribution.
        """

        return sum(self.log_likelihood(s) for s in samples)

    def __get_ndomains(self):
        """
        Return the number of mixture component domains in this distribution.
        """

        return self.__components_MK.shape[0]

    def __get_ncomponents(self):
        """
        Return the number of mixture components in this distribution.
        """

        return self.__components_MK.shape[1]

    def __get_pi(self):
        return self.__pi_K

    def __get_components(self):
        return self.__components_MK

    def __get_shapes(self):
        """
        Return the shapes of component distributions.
        """

        return self.__shapes

    # properties
    ndomains = property(__get_ndomains)
    ncomponents = property(__get_ncomponents)
    pi = property(__get_pi)
    components = property(__get_components)
    shapes = property(__get_shapes)

class FixedIndicatorMixtureEstimator(object):
    """
    Estimate the parameters of a finite [linked] mixture distribution.
    """

    def __init__(self, estimators, pi_K = None):
        """
        Initialize.

        @param estimators: A list of [lists of] estimators of the component distributions.
        """

        # estimators
        estimators_MK = numpy.asarray(estimators)

        if estimators_MK.ndim == 1:
            self.__estimators_MK = estimators_MK[newaxis, :]
        elif estimators_MK.ndim == 2:
            self.__estimators_MK = estimators_MK
        else:
            raise ArgumentError("estimator list must be one- or two-dimensional")

        (_, K) = estimators_MK.shape

        # fixed pi
        if pi_K is None:
            self.pi_K = numpy.ones(K) / K
        else:
            self.pi_K = pi_K

    def estimate(self, samples, verbose = False):
        """
        Use EM to estimate DCM mixture parameters.
        """

        # FIXME support more than one sample

        # mise en place
        (M, K) = self.__estimators_MK.shape
        float_finfo = numpy.finfo(numpy.float)

        assert len(samples) == M

        for m in xrange(M):
            assert samples[m].shape[0] == K

        # generate random initial component parameterizations
        components_MK = numpy.empty((M, K), numpy.object)

        for m in xrange(M):
            for k in xrange(K):
                components_MK[m, k] = self.__estimators_MK[m, k].estimate(numpy.array([samples[m][k]]))

        return FiniteMixture(self.pi_K, components_MK)

class ExpectationMaximizationMixtureEstimator(object):
    """
    Estimate the parameters of a finite [linked] mixture distribution using EM.
    """

    def __init__(self, estimators):
        """
        Initialize.

        @param estimators: A list of [lists of] estimators of the component distributions.
        """

        # estimators
        estimators_MK = numpy.asarray(estimators)

        if estimators_MK.ndim == 1:
            self.__estimators_MK = estimators_MK[newaxis, :]
        elif estimators_MK.ndim == 2:
            self.__estimators_MK = estimators_MK
        else:
            raise ArgumentError("estimator list must be one- or two-dimensional")

        # other parameters
        self.__iterations = 24 # FIXME

    def estimate(self, samples, verbose = False):
        """
        Use EM to estimate DCM mixture parameters.
        """

        # FIXME could initialize more intelligently
        # FIXME need a non-lame termination criterion

        # mise en place
        (M, K) = self.__estimators_MK.shape
        float_finfo = numpy.finfo(numpy.float)

        assert len(samples) == M

        N = samples[0].shape[0]

        for m in xrange(M):
            assert samples[m].shape[0] == N

        pi_K = numpy.random.random(K)
        pi_K /= numpy.sum(pi_K)

        # generate random initial component parameterizations
        components_MK = numpy.empty((M, K), numpy.object)

        for m in xrange(M):
            for k in xrange(K):
                weights = numpy.random.random(N)

#                log.debug("%i.%i (initializing)", m, k)

                components_MK[m, k] = self.__estimators_MK[m, k].estimate(samples[m], weights = weights)

#        log.debug("done initializing")

        # take some number of EM steps
        for i in xrange(self.__iterations):
            # evaluate the responsibilities
            r_NK = numpy.empty((N, K))

            for n in xrange(N):
                for k in xrange(K):
                    r = pi_K[k]

                    for m in xrange(M):
#                        log.debug("%s", str(components_MK[m, k].alpha))

                        r *= numpy.exp(components_MK[m, k].log_likelihood(samples[m][n]))

                    if r == 0.0:
                        r_NK[n, k] = float_finfo.tiny
                    else:
                        r_NK[n, k] = r

            r_NK /= numpy.sum(r_NK, 1)[:, newaxis]

            # find the maximum-likelihood estimates of components
            for m in xrange(M):
                for k in xrange(K):
#                    log.debug("%i.%i (iteration %i)", m, k, i)

                    components_MK[m, k] = self.__estimators_MK[m, k].estimate(samples[m], weights = r_NK[:, k])

            # find the maximum-likelihood pis
            pi_K = numpy.sum(r_NK, 0) / N

            # tracing
            com = numpy.sum((numpy.arange(K) + 1) * pi_K)

            log.info("EM iteration %i of %i: pi com = %f", i + 1, self.__iterations, com)

        # done
        log.info("pi %s (sum %f)", str(pi_K), numpy.sum(pi_K))

        return FiniteMixture(pi_K, components_MK)

