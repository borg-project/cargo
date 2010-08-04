"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy                      import newaxis
from cargo.log                  import get_logger
from cargo.statistics.base      import (
    Estimator,
    Distribution,
    )
from cargo.statistics.functions import add_log

log = get_logger(__name__)

class FiniteMixture(Distribution):
    """
    An arbitrary finite [linked] mixture distribution.
    """

    def __init__(self, pi, components):
        """
        Initialize.
        """

        # sanity
        if len(pi) != len(components):
            raise ValueError("component and parameter counts do not match")

        # members
        from cargo.statistics.discrete import ObjectDiscrete

        self._mixer = ObjectDiscrete(pi, components)

    def random_variate(self, random = numpy.random):
        """
        Make a draw from this mixture distribution.
        """

        return self._mixer.random_variate(random).random_variate(random)

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        from itertools import izip

        total = numpy.finfo(float).min

        for (pi_k, component) in izip(self.pi, self.components):
            total = add_log(total, numpy.log(pi_k))
            total = add_log(total, component.log_likelihood(sample))

        return total

    @property
    def pi(self):
        """
        Return the mixture parameter vector.
        """

        return self._mixer.beta

    @property
    def components(self):
        """
        Return the mixture components.
        """

        return self._mixer.domain

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

    def estimate(self, samples, random = numpy.random):
        """
        Make multiple estimates, and return the best.
        """

        sane_samples = zip(*samples)

        best_ll       = None
        best_estimate = None

        for i in xrange(self._nrestarts):
            estimate = self._estimator.estimate(samples, random = random)
            ll       = estimate.total_log_likelihood(sane_samples)

            if best_ll is None:
                log.info("l-l of estimate is %e", ll)
            else:
                log.info("l-l of estimate is %e (best is %e)", ll, best_ll)

            if best_ll is None or ll > best_ll:
                best_ll       = ll
                best_estimate = estimate

        return best_estimate

