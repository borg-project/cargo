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
    npy_intp,
    PyUFunc_None,
    PyUFuncGenericFunction,
    PyUFunc_FromFuncAndData,
    import_array,
    import_ufunc,
    )

import_ufunc()
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

cdef void mixed_binomial_ll_ufunc_loop(char** args, npy_intp* dimensions, npy_intp* strides, void* data):
    """
    The core loop of the mixed-binomial log-probability ufunc.
    """

    #cdef char* p        = args[0]
    #cdef char* s        = args[1]
    #cdef char* o        = args[2]
    #cdef int   p_stride = strides[0]
    #cdef int   s_stride = strides[1]
    #cdef int   o_stride = strides[2]
    #cdef size_t i

    #for i in xrange(dimensions[0]):
        ##out[i] = log(scipy.stats.binom.pmf(samples[i].k, samples[i].n, parameters[i]))
        ##(<double*>o)[0] = (<double*>s)[0] + (<double*>p)[0]

        #p += p_stride
        #s += s_stride
        #o += o_stride

def mixed_binomial_ll_ufunc():
    """
    Construct the mixed-binomial log-probability ufunc.
    """

    # allocate the functions array
    cdef PyUFuncGenericFunction* functions = <PyUFuncGenericFunction*>malloc(sizeof(PyUFuncGenericFunction) * 1)

    functions[0] = mixed_binomial_ll_ufunc_loop

    # allocate the data array
    cdef void** data = <void**>malloc(sizeof(void*) * 1)

    data[0] = NULL

    # construct the ufunc
    return \
        PyUFunc_FromFuncAndData(
            functions,                           # functions
            data,                                # data
            "ddd",                               # types
            1,                                   # ntypes
            2,                                   # nin
            1,                                   # nout
            PyUFunc_None,                        # identity
            "binomial_ll",                       # name
            "Compute binomial log-probability.", # doc
            0,                                   # check_return
            )

ll = mixed_binomial_ll_ufunc()

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

    #def ll(
                                               #self,
        #ndarray[float_t            , ndim = 2] parameters,
        #ndarray[MixedBinomialSample, ndim = 2] samples,
        #ndarray[float_t            , ndim = 2] out = None,
        #):
        #"""
        #Return the log probability of samples under this distribution.
        #"""

        ## arguments
        #assert samples.shape[0] == parameters.shape[0]

        #if out is None:
            #out = numpy.empty((samples.shape[0], samples.shape[1]), numpy.float_)
        #else:
            #assert samples.shape[0] == out.shape[0]
            #assert samples.shape[1] == out.shape[1]

        ## computation
        #cdef size_t              i
        #cdef size_t              j
        #cdef MixedBinomialSample s

        #for i in xrange(out.shape[0]):
            #for j in xrange(out.shape[1]):
                #s         = samples[i, j]
                #out[i, j] = scipy.stats.binom.pmf(s.k, s.n, parameters[i, j])

        #numpy.log(out, out)

        #return out

    def ml(
                                               self,
        ndarray[MixedBinomialSample, ndim = 2] samples,
        ndarray[float_t            , ndim = 2] weights,
        ndarray[float_t            , ndim = 1] out = None,
                                               random = numpy.random,
        ):
        """
        Return the estimated maximum-likelihood parameter.
        """

        # arguments
        assert samples.shape[0] == weights.shape[0]
        assert samples.shape[1] == weights.shape[1]

        if out is None:
            out = numpy.empty(samples.shape[0], numpy.float_)
        else:
            assert samples.shape[0] == out.shape[0]

        # computation
        cdef uint_t              i
        cdef uint_t              j
        cdef float_t             k
        cdef float_t             n
        cdef MixedBinomialSample s

        for i in xrange(samples.shape[0]):
            k = 0.0
            n = 0.0

            for j in xrange(samples.shape[1]):
                s  = samples[i, j]
                k += s.k * weights[i, j]
                n += s.n * weights[i, j]

            out[i] = (k + self._epsilon) / (n + self._epsilon)

        return out

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

