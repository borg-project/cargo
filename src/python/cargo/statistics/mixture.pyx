# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.log import get_logger

cimport numpy

from numpy cimport (
    ndarray,
    float_t,
    )

log = get_logger(__name__)

class FiniteMixture(Distribution):
    """
    An arbitrary finite homogeneous mixture distribution.
    """

    def __init__(self, distribution, K, iterations = 256, convergence = 1e-8):
        """
        Initialize.
        """

        self._distribution    = distribution
        self._K               = K
        self._iterations      = iterations
        self._convergence     = convergence
        self._parameter_dtype = \
            numpy.dtype((
                [
                    ("p", numpy.float_),
                    ("c", distribution.parameter_dtype),
                    ],
                K,
                ))

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

    def ml(
                                   self,
        ndarray[         ndim = 1] samples,
        ndarray[float_t, ndim = 1] weights,
                                   random = numpy.random,
        ):
        """
        Use EM to estimate mixture parameters.
        """

        if weights is not None:
            raise NotImplementedError("sample weighting not yet supported")

        log.detail("estimating finite mixture from %i samples" % len(samples))

        # generate random initial parameters
        from cargo.random import grab

        one_weights = numpy.ones_like(samples)
        components  = [d.ml([grab(samples, random)], one_weights, random) for d in self._distributions]

        pi_K  = random.rand(len(self._distributions))
        pi_K /= numpy.sum(pi_K)

        # run EM until convergence
        last_r_NK = None
        r_NK      = numpy.empty((len(samples), len(components)))
        ll_N      = numpy.empty(len(samples))

        for i in xrange(self._iterations):
            # evaluate the responsibilities
            r_NK[:, :] = 0.0

            for (k, distribution) in enumerate(self._distributions):
                distribution.ll(components[k], samples, ll_N)

                r_NK[:, k] += ll_N

            numpy.exp(r_NK, r_NK)

            r_NK *= pi_K[None, :]
            r_NK /= numpy.sum(r_NK, 1)[:, None]

            # find the maximum-likelihood estimates of components
            for (k, distribution) in enumerate(self._distributions):
                components[k] = distribution.ml(samples, r_NK[:, k], random)

            # find the maximum-likelihood pis
            pi_K = numpy.sum(r_NK, 0) / len(samples)

            # termination?
            if last_r_NK is None:
                last_r_NK = numpy.empty((len(samples), len(components)))
            else:
                difference = numpy.sum(numpy.abs(r_NK - last_r_NK))

                log.detail(
                    "iteration %i < %i ; delta %e >? %e",
                    i,
                    self._iterations,
                    difference,
                    self._convergence,
                    )

            (last_r_NK, r_NK) = (r_NK, last_r_NK)

        # done
        return FiniteMixture(pi_K, components)

    @property
    def distribution(self):
        """
        Return the mixture components.
        """

        return self._distribution

    @property
    def parameter_dtype(self):
        """
        Return the parameter type.
        """

        return self._parameter_dtype

    def 

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

