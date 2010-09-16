# cython: profile=True
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
            if pi_k > 0.0:
                total = add_log(total, numpy.log(pi_k) + component.log_likelihood(sample))

        return total

    def given(self, samples):
        """
        Return the conditional distribution.
        """

        # generate the posterior mixture parameters
        post_pi = numpy.copy(self.pi)

        for k in xrange(post_pi.size):
            post_pi[k] *= numpy.exp(self.components[k].total_log_likelihood(samples))

        post_pi /= numpy.sum(post_pi)

        # generate the posterior mixture components
        post_components = [c.given(samples) for c in self.components]

        # done
        return FiniteMixture(post_pi, post_components)

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

    def __init__(self, estimators, iterations = 8, convergence = 1e-8):
        """
        Initialize.

        @param estimators: Estimators of the component distributions.
        """

        self._estimators  = estimators
        self._iterations  = iterations
        self._convergence = convergence

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Use EM to estimate mixture parameters.
        """

        if weights is not None:
            raise NotImplementedError("sample weighting not yet supported")

        log.detail("estimating finite mixture from %i samples" % len(samples))

        # generate random initial parameters
        from cargo.random import grab

        components = [e.estimate([grab(samples, random)]) for e in self._estimators]

        pi_K  = random.rand(len(self._estimators))
        pi_K /= numpy.sum(pi_K)

        # run EM until convergence
        last_r_NK = None
        r_NK      = numpy.empty((len(samples), len(components)))

        for i in xrange(self._iterations):
            # evaluate the responsibilities
            r_NK[:, :] = 0.0

            for (k, component) in enumerate(components):
                component.add_log_likelihoods(samples, r_NK[:, k])

            numpy.exp(r_NK, r_NK)

            r_NK *= pi_K[None, :]
            r_NK /= numpy.sum(r_NK, 1)[:, None]

            # find the maximum-likelihood estimates of components
            for (k, estimator) in enumerate(self._estimators):
                components[k] = estimator.estimate(samples, random, r_NK[:, k])

            # find the maximum-likelihood pis
            pi_K = numpy.sum(r_NK, 0) / len(samples)

            # tracing
            log.debug(
                "pi [%s] (com %.2f)",
                " ".join(["%.2f" % p for p in pi_K]),
                numpy.sum((numpy.arange(len(components)) + 1) * pi_K),
                )

            # termination?
            if last_r_NK is None:
                last_r_NK = numpy.empty((len(samples), len(components)))
            else:
                difference = numpy.sum(numpy.abs(r_NK - last_r_NK))

                if difference < self._convergence:
                    log.detail("converged with delta_r = %e", difference)

                    break
                else:
                    log.detail(
                        "iteration %i < %i: %e to convergence",
                        i,
                        self._iterations,
                        difference - self._convergence,
                        )

            (last_r_NK, r_NK) = (r_NK, last_r_NK)

        # done
        return FiniteMixture(pi_K, components)

    @property
    def estimators(self):
        """
        Return the component estimators.
        """

        return self._estimators

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

