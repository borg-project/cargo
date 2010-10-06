# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy.stats

cimport numpy
cimport cython

from libc.stdlib cimport malloc
from numpy       cimport (
    ndarray,
    uint_t,
    float_t,
    import_array,
    broadcast,
    PyArray_MultiIter_DATA,
    PyArray_MultiIter_NEXT,
    PyArray_MultiIter_NOTDONE,
    )

import_array()

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

        numpy.log(out, out)

        return out

cdef packed struct MixedBinomialSample:
    uint_t k
    uint_t n

class MixedBinomial(object):
    """
    Operate on multiple binomials with a single common probability parameter.
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

    def ll(self, parameters, samples, out = None):
        """
        Compute the log probabilities of binomial samples.
        """

        # arguments
        parameters = numpy.asarray(parameters, self._parameter_dtype)
        samples    = numpy.asarray(samples, self._sample_dtype)

        cdef broadcast i = broadcast(parameters, samples)

        if out is None:
            out = numpy.empty(i.shape)
        else:
            if out.shape != i.shape:
                raise ValueError("out argument has invalid shape")
            if out.dtype != numpy.float_:
                raise ValueError("out argument has invalid dtype")

        # computation
        i = broadcast(parameters, samples, out)

        cdef double              p
        cdef double              v
        cdef MixedBinomialSample s

        while PyArray_MultiIter_NOTDONE(i):
            p = (<double*>PyArray_MultiIter_DATA(i, 0))[0]
            s = (<MixedBinomialSample*>PyArray_MultiIter_DATA(i, 1))[0]
            v = numpy.log(scipy.stats.binom.pmf(s.k, s.n, p))

            (<double*>PyArray_MultiIter_DATA(i, 2))[0] = v

            PyArray_MultiIter_NEXT(i)

        return out

    def ml(self, samples, weights, out = None, random = numpy.random):
        """
        Return the estimated maximum-likelihood parameter.
        """

        # arguments
        (broad_samples, broad_weights) = \
            numpy.broadcast_arrays(
                numpy.asarray(samples, self._sample_dtype),
                numpy.asarray(weights),
                )

        # restore the dtype (broadcast_arrays partially strips it)
        cdef ndarray[MixedBinomialSample, ndim = 2] samples_ = numpy.asarray(broad_samples, self._sample_dtype)
        cdef ndarray[float_t            , ndim = 2] weights_ = broad_weights
        cdef ndarray[float_t            , ndim = 1] out_

        if out is None:
            out_ = numpy.empty(samples_.shape[0], numpy.float_)
        else:
            out_ = out

            assert out_.ndim == 1
            assert samples_.shape[0] == out_.shape[0]

        # computation
        cdef uint_t              i
        cdef uint_t              j
        cdef float_t             k
        cdef float_t             n
        cdef MixedBinomialSample s

        for i in xrange(samples_.shape[0]):
            k = 0.0
            n = 0.0

            for j in xrange(samples_.shape[1]):
                s  = samples_[i, j]
                k += s.k * weights_[i, j]
                n += s.n * weights_[i, j]

            out_[i] = (k + self._epsilon) / (n + self._epsilon)

        return out_

    @property
    def parameter_dtype(self):
        """
        Type of a distribution parameter.
        """

        return self._parameter_dtype

    @property
    def sample_dtype(self):
        """
        Type of a sample.
        """

        return self._sample_dtype

