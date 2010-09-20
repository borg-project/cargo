# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

cimport numpy

from numpy cimport (
    ndarray,
    uint_t,
    float_t,
    )

class Tuple(Distribution):
    """
    A tuple of independent distributions.
    """

    def __init__(self, distributions):
        """
        Initialize.
        """

        self._distributions   = distributions
        self._names           = ["d%i" % i for i in xrange(len(distributions))]
        self._parameter_dtype = \
            numpy.dtype([
                (n, d.parameter_dtype)
                for n in self._names
                for d in distributions
                ])
        self._sample_dtype    = \
            numpy.dtype([
                (n, d.sample_dtype)
                for n in self._names
                for d in distributions
                ])

    def rv(
                          self,
        ndarray[ndim = 1] parameters,
        ndarray[ndim = 2] out,
                          random = numpy.random,
        ):
        """
        Return samples from this distribution.
        """

        # arguments
        assert parameters.shape[0] == out.shape[0]

        # computation
        cdef size_t            i
        cdef size_t            j
        cdef BinomialParameter p

        for i in xrange(out.shape[0]):
            p         = parameters[i]
            out[i, :] = random.binomial(p.n, p.p, out.shape[1])

        return out

    def ll(
                          self,
        ndarray[ndim = 1] parameters,
        ndarray[ndim = 2] samples,
        ndarray[ndim = 2] out = None,
        ):
        """
        Return the log probability of samples under this distribution.
        """

        # arguments
        assert len(parameters.dtype.names) == len(self._distributions)
        assert samples.shape[0] == parameters.shape[0]

        if out is None:
            out = numpy.empty((samples.shape[0], samples.shape[1]), numpy.float_)
        else:
            assert samples.shape[0] == out.shape[0]
            assert samples.shape[1] == out.shape[1]

        # computation
        for (name, distribution) in zip(self._names, self._distributions):
            distribution.ll(parameters[name], samples[name], out[name])

        return out

    def ml(
                                   self,
        ndarray[         ndim = 1] samples,
        ndarray[float_t, ndim = 1] weights,
                                   random = numpy.random,
        ):
        """
        Return the estimated maximum-likelihood parameter.
        """

        #parameters = numpy.

    @property
    def distributions(self):
        """
        Return the inner distributions.
        """

        return self._distributions

    @property
    def parameter_dtype(self):
        """
        Type of distribution parameter(s).
        """

        return self._parameter_dtype

    @property
    def sample_dtype(self):
        """
        Type of distribution parameter(s).
        """

        return self._sample_dtype

class TupleEstimator(Estimator):
    """
    A maximum-likelihood estimator of tuple distributions.
    """

    def __init__(self, estimators):
        """
        Initialize.
        """

        self._estimators = estimators

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        if not isinstance(samples, TupleSamples):
            samples = TupleSamples.from_sequence(samples)

        if len(samples._sequences) != len(self._estimators):
            raise \
                ValueError(
                    "samples width %i does not match estimators count %i" % (
                        len(samples._sequences),
                        len(self._estimators),
                        ),
                    )

        from itertools import izip

        zipped = izip(self._estimators, samples._sequences)

        return TupleDistribution([e.estimate(s, random, weights) for (e, s) in zipped])

