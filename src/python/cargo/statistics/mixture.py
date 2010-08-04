"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log             import get_logger
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

log = get_logger(__name__)

class FiniteMixture(Distribution):
    """
    An arbitrary finite mixture distribution.

    Relevant types:
        - sample: arbitrary; sampled from a component
        - sequence: list
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

        from itertools                  import izip
        from cargo.statistics.functions import add_log

        total = numpy.finfo(float).min

        for (pi_k, component) in izip(self.pi, self.components):
            total = add_log(total, numpy.log(pi_k) + component.log_likelihood(sample))

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

class EM_MixtureEstimator(Estimator):
    """
    Estimate the parameters of a finite mixture distribution using EM.
    """

    def __init__(self, estimators):
        """
        Initialize.

        @param estimators: Estimators of the component distributions.
        """

        self._estimators  = estimators
        self._iterations  = 32
        self._convergence = 1e-8

    def estimate(self, samples, random = numpy.random):
        """
        Use EM to estimate mixture parameters.
        """

        # generate random initial parameters
        from cargo.random import grab

        components = [e.estimate(grab(samples, random)) for e in self._estimators]

        pi_K  = random.rand(len(self._estimators))
        pi_K /= numpy.sum(pi_K)

        # run EM until convergence
        from numpy import newaxis

        last_r_NK = None
        r_NK      = numpy.empty((len(samples), len(components)))

        for i in xrange(self._iterations):
            # evaluate the responsibilities
            for (k, component) in enumerate(components):
                for (n, sample) in enumerate(samples):
                    r  = pi_K[k]
                    r *= numpy.exp(component.log_likelihood(sample))

                    if r == 0.0:
                        r_NK[n, k] = numpy.finfo(numpy.float).tiny
                    else:
                        r_NK[n, k] = r

            r_NK /= numpy.sum(r_NK, 1)[:, newaxis]

            # find the maximum-likelihood estimates of components
            for (k, estimator) in enumerate(self._estimators):
                components[k] = estimator.estimate(samples, random, r_NK[:, k])

            # find the maximum-likelihood pis
            pi_K = numpy.sum(r_NK, 0) / len(samples)

            # tracing
            log.debug(
                "pi [%s] (com %.2f)",
                " ".join("%.2f" % p for p in pi_K),
                numpy.sum((numpy.arange(len(components)) + 1) * pi_K),
                )

            # termination?
            if last_r_NK is None:
                last_r_NK = numpy.empty((len(samples), len(components)))
            else:
                difference = numpy.sum(numpy.abs(r_NK - last_r_NK))

                log.detail("delta_r = %e", difference)

                if difference < self._convergence:
                    break

            (last_r_NK, r_NK) = (r_NK, last_r_NK)

        # done
        return FiniteMixture(pi_K, components)

class RestartedEstimator(Estimator):
    """
    Make multiple estimates, and return the best.
    """

    def __init__(self, estimator, restarts = 2):
        """
        Initialize.
        """

        self._estimator = estimator
        self._restarts  = restarts

    def estimate(self, samples, random = numpy.random):
        """
        Make multiple estimates, and return the apparent best.
        """

        best_ll       = None
        best_estimate = None

        for i in xrange(self._restarts):
            estimated = self._estimator.estimate(samples, random = random)
            ll        = estimated.total_log_likelihood(samples)

            if best_ll is None or ll > best_ll:
                best_ll       = ll
                best_estimate = estimated

            log.info("l-l of estimate is %e (best is %e)", ll, best_ll)

        return best_estimate

