# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

cimport numpy
cimport cython

from numpy cimport (
    ndarray,
    uint_t,
    float_t,
    )

cdef packed struct BinomialParameter:
    float_t p
    uint_t  n

class Binomial(object):
    """
    Operations of the binomial distribution.

    Relevant types:
    - sample    : uint
    - samples   : N-d uint array
    - parameter : 0-d [("p", float), ("n", uint)] array
    """

    def rv(
                                             self,
        ndarray[BinomialParameter, ndim = 1] parameters,
        ndarray[uint_t           , ndim = 2] out,
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
        ndarray[BinomialParameter, ndim = 1] parameters,
        ndarray[uint_t           , ndim = 2] samples,
        ndarray[float_t          , ndim = 2] out = None,
        ):
        """
        Return the log probability of C{sample} under this distribution.
        """

        # arguments
        assert samples.shape[0] == parameters.shape[0]

        if out is None:
            out = numpy.empty((samples.shape[0], samples.shape[1]), numpy.float_)
        else:
            assert samples.shape[0] == out.shape[0]
            assert samples.shape[1] == out.shape[1]

        # computation
        cdef size_t            i
        cdef size_t            j
        cdef BinomialParameter p

        for i in xrange(out.shape[0]):
            p = parameters[i]

            for j in xrange(out.shape[1]):
                out[i, j] = scipy.stats.binom.pmf(samples[i, j], p.n, p.p)

        return out

cdef packed struct MixedBinomialSample:
    uint_t k
    uint_t n

class MixedBinomial(object):
    """
    Operations of the binomial distribution in a typical mixture.
    """

    _parameter_dtype = numpy.float_
    _sample_dtype    = numpy.dtype([("k", numpy.uint), ("n", numpy.uint)])

    def __init__(self, epsilon = 1e-3):
        """
        Initialize.
        """

        if epsilon is None:
            self._epsilon = 0.0
        else:
            self._epsilon = epsilon

    def ll(
                                               self,
        ndarray[float_t            , ndim = 1] parameters,
        ndarray[MixedBinomialSample, ndim = 2] samples,
        ndarray[float_t            , ndim = 2] out = None,
        ):
        """
        Return the log probability of samples under this distribution.
        """

        # arguments
        assert samples.shape[0] == parameters.shape[0]

        if out is None:
            out = numpy.empty((samples.shape[0], samples.shape[1]), numpy.float_)
        else:
            assert samples.shape[0] == out.shape[0]
            assert samples.shape[1] == out.shape[1]

        # computation
        cdef size_t              i
        cdef size_t              j
        cdef MixedBinomialSample s

        for i in xrange(out.shape[0]):
            for j in xrange(out.shape[1]):
                s         = samples[i, j]
                out[i, j] = scipy.stats.binom.pmf(s.k, s.n, parameters[i])

        return out

    def ml(
                                               self,
        ndarray[MixedBinomialSample, ndim = 1] samples,
        ndarray[float_t            , ndim = 1] weights,
                                               random = numpy.random,
        ):
        """
        Return the estimated maximum-likelihood parameter.
        """

        cdef uint_t  i
        cdef float_t k = 0
        cdef float_t n = 0

        for i in xrange(samples.shape[0]):
            k += samples[i].k * weights[i]
            n += samples[i].n * weights[i]

        return (k + self._epsilon) / (n + self._epsilon)

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

