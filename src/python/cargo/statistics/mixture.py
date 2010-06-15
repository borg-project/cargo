"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy                      import newaxis
from cargo.log                  import get_logger
from cargo.statistics.functions import add_log

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
        self.__pi_K  = numpy.asarray(pi)

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

        random  = kwargs.get("random", numpy.random)
        (M, K)  = self.__components_MK.shape
        ((k,),) = numpy.nonzero(random.multinomial(1, self.__pi_K))

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

        @param samples: List of samples from each of the domains.
        """

         # FIXME surely we can do better numerically here

        # parameters and sanity
        (M, K) = self.__components_MK.shape

        assert len(samples) == M

        # draw the sample(s)
#         total = 0.0
        total = None

        for k in xrange(K):
#             ctotal = 0.0
            ctotal = self.__pi_K[k]

            for m in xrange(M):
                sample_l_l = self.__components_MK[m, k].log_likelihood(samples[m])

                if sample_l_l > 0.0:
                    log.warning("positive l-l for %s", samples[m])

                ctotal += sample_l_l

                if not numpy.isfinite(ctotal):
#                     log.debug("ctotal %s; component (%i) %s", ctotal, m, self.__components_K[m, k].log_beta)
#                     log.debug("and beta is... %s", self.__components_MK[m, k].beta)
                    log.warning("nonfinite ctotal %s", ctotal)
                    log.warning("samples[m] is %s", samples[m])

#             total += self.__pi_K[k] * numpy.exp(ctotal)
            if total is None:
                total = ctotal
            else:
                total = add_log(total, ctotal)

            log.debug("%s ; %s (%i of %i)", ctotal, total, k, K)

#         if total == 0.0:
        if not numpy.isfinite(total):
            log.warning("sample has zero probability")
            log.debug("the sample in question: %s", repr(samples))

#         return numpy.log(total)
        return total

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of many C{samples} lists under this distribution.

        @param samples: List of lists of samples from each of the domains.
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
    Estimate the parameters of a finite [linked] mixture distribution 
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

    def estimate(self, samples):
        """
        Use EM to estimate mixture parameters.
        """

        # mise en place
        (M, K) = self.__estimators_MK.shape
        float_finfo = numpy.finfo(numpy.float)

        assert len(samples) == M

        N = samples[0].shape[0]

        for m in xrange(M):
            assert samples[m].shape[0] == N

        pi_K  = numpy.ones(K)
        pi_K /= numpy.sum(pi_K)

        # generate random initial component parameterizations
        components_MK = numpy.empty((M, K), numpy.object)

        for k in xrange(K):
            for m in xrange(M):
                components_MK[m, k] = self.__estimators_MK[m, k].estimate(samples[m][k:k + 1])

        # done
        return FiniteMixture(pi_K, components_MK)

class EM_MixtureEstimator(object):
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

        # other members
        self.__max_i       = 32
        self.__convergence = 1e-8

    def estimate(self, samples, random = numpy.random):
        """
        Use EM to estimate mixture parameters.
        """

        # mise en place
        (M, K) = self.__estimators_MK.shape
        float_finfo = numpy.finfo(numpy.float)

        assert len(samples) == M

        N = samples[0].shape[0]

        for m in xrange(M):
            assert samples[m].shape[0] == N

        pi_K  = numpy.ones(K)
        pi_K /= numpy.sum(pi_K)

        # generate random initial component parameterizations
        components_MK = numpy.empty((M, K), numpy.object)

        for k in xrange(K):
            n = random.randint(N)

            for m in xrange(M):
                components_MK[m, k] = self.__estimators_MK[m, k].estimate(samples[m][n:n + 1])

        # run EM until convergence
        previous_r_NK = None

        for i in xrange(self.__max_i):
#             for k in xrange(K):
#                 log.info("% 2s: %s (%.2f)", k, " ".join("%.2f" % c.beta[0] for c in components_MK[:, k]), pi_K[k])

            # evaluate the responsibilities
            r_NK = numpy.empty((N, K))

            for k in xrange(K):
                for n in xrange(N):
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
            for k in xrange(K):
                for m in xrange(M):
                    components_MK[m, k] = self.__estimators_MK[m, k].estimate(samples[m], weights = r_NK[:, k])

            # find the maximum-likelihood pis
            pi_K = numpy.sum(r_NK, 0) / N

            # tracing
            log.debug(
                "pi [%s] (com %.2f)",
                " ".join("%.2f" % p for p in pi_K),
                numpy.sum((numpy.arange(K) + 1) * pi_K),
                )

            # termination?
            if previous_r_NK is not None:
                difference = numpy.sum(numpy.abs(r_NK - previous_r_NK))

                if difference < self.__convergence:
                    log.detail("difference in responsibilities is %e; converged", difference)

                    break
                else:
                    log.detail("difference in responsibilities is %e; not converged", difference)

            previous_r_NK = r_NK

        # done
        return FiniteMixture(pi_K, components_MK)

class RestartedEstimator(object):
    """
    Make multiple estimates, and return the best.
    """

    def __init__(self, estimator, nrestarts = 2):
        """
        Initialize.
        """

        self._estimator = estimator
        self._nrestarts = nrestarts

    def estimate(self, samples):
        """
        Use EM to estimate mixture parameters.
        """

        # FIXME our choice of structure for samples is stupid
        sane_samples = zip(*samples)

        best_ll       = None
        best_estimate = None

        for i in xrange(self._nrestarts):
            estimate = self._estimator.estimate(samples)
            ll       = estimate.total_log_likelihood(sane_samples)

            if best_ll is None:
                log.info("l-l of estimate is %e", ll)
            else:
                log.info("l-l of estimate is %e (best is %e)", ll, best_ll)

            if best_ll is None or ll > best_ll:
                best_ll       = ll
                best_estimate = estimate

        return best_estimate

