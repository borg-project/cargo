"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from llvm.core import (
    Type,
    Constant,
    )

class Binomial(object):
    """
    Build low-level operations of the binomial distribution.

    Relevant types:
    - parameter : {f64 p; u32 n;}
    - sample    : u32
    """

    def __init__(self):
        """
        Initialize.
        """

        self._parameter_t = Type.packed_struct([Type.double(), Type.int(32)])
        self._sample_t    = Type.int(32)

    def for_module(self, module):
        """
        Return core for use in a specific module.
        """

        return BinomialBuilder(module)

    @property
    def parameter_type(self):
        """
        LLVM type of the distribution parameter.
        """

        return self._parameter_t

    @property
    def sample_type(self):
        """
        LLVM type of the distribution sample.
        """

        return self._sample_t

class BinomialBuilder(object):
    """
    Build low-level operations of the binomial distribution.
    """

    def __init__(self, module):
        """
        Initialize.
        """

        print "wtf?"

        # link the GSL
        import ctypes

        from ctypes      import CDLL
        from ctypes.util import find_library

        ctypes.CDLL(find_library("cblas"), ctypes.RTLD_GLOBAL)
        ctypes.CDLL(find_library("gsl"  ), ctypes.RTLD_GLOBAL)

        # set up the builder
        from ctypes    import sizeof

        self._ln_function = \
            module.add_function(
                Type.function(Type.double(), [Type.double()]),
                "log",
                )
        self._ll_function = \
            module.add_function(
                Type.function(
                    Type.double(),
                    [
                        Type.int(sizeof(ctypes.c_uint) * 8),
                        Type.double(),
                        Type.int(sizeof(ctypes.c_uint) * 8),
                        ],
                    ),
                "gsl_ran_binomial_pdf",
                )

    def rv(self, builder, parameter, random):
        """
        Return samples from this distribution.
        """

        raise NotImplementedError()

    def ll(self, builder, parameter, sample):
        """
        Compute log probability under this distribution.
        """

        return \
            builder.call(
                self._ln_function,
                [
                    builder.call(
                        self._ll_function,
                        [
                            sample,
                            builder.getresult(parameter, 0),
                            builder.getresult(parameter, 1),
                            ],
                        ),
                    ],
                )

    def ml(self, sam_loop, weight_loop, out_p, prng):
        """
        Return the estimated maximum-likelihood parameter.
        """

        raise NotImplementedError()

#cdef packed struct MixedBinomialSample:
    #uint_t k
    #uint_t n

#class MixedBinomial(object):
    #"""
    #Operate on multiple binomials with a single common probability parameter.
    #"""

    #_parameter_dtype = numpy.dtype(numpy.float_)
    #_sample_dtype    = numpy.dtype([("k", numpy.uint), ("n", numpy.uint)])

    #def __init__(self, epsilon = 1e-3):
        #"""
        #Initialize.
        #"""

        #if epsilon is None:
            #self._epsilon = 0.0
        #else:
            #self._epsilon = epsilon

    #def ll(self, parameters, samples, out = None):
        #"""
        #Compute binomial log-likelihood.
        #"""

        ## arguments
        #parameters = numpy.asarray(parameters, self._parameter_dtype)
        #samples    = numpy.asarray(samples, self._sample_dtype)

        #cdef broadcast i = broadcast(parameters, samples)

        #if out is None:
            #out = numpy.empty(i.shape)
        #else:
            #if out.shape != i.shape:
                #raise ValueError("out argument has invalid shape")
            #if out.dtype != numpy.float_:
                #raise ValueError("out argument has invalid dtype")

        ## computation
        #i = broadcast(parameters, samples, out)

        #cdef double              p
        #cdef double              v
        #cdef MixedBinomialSample s

        #while PyArray_MultiIter_NOTDONE(i):
            #p = (<double*>PyArray_MultiIter_DATA(i, 0))[0]
            #s = (<MixedBinomialSample*>PyArray_MultiIter_DATA(i, 1))[0]
            #v = log(binomial_pdf(s.k, p, s.n))

            #(<double*>PyArray_MultiIter_DATA(i, 2))[0] = v

            #PyArray_MultiIter_NEXT(i)

        #return out

    #def ml(self, samples, weights, out = None, random = numpy.random):
        #"""
        #Return the estimated maximum-likelihood parameter.
        #"""

        ## arguments
        #(broad_samples, broad_weights) = \
            #numpy.broadcast_arrays(
                #numpy.asarray(samples, self._sample_dtype),
                #numpy.asarray(weights),
                #)

        ## restore the dtype (broadcast_arrays partially strips it)
        #cdef ndarray[MixedBinomialSample, ndim = 2] samples_ = numpy.asarray(broad_samples, self._sample_dtype)
        #cdef ndarray[float_t            , ndim = 2] weights_ = broad_weights
        #cdef ndarray[float_t            , ndim = 1] out_

        #if out is None:
            #out_ = numpy.empty(samples_.shape[0], numpy.float_)
        #else:
            #out_ = out

            #assert out_.ndim == 1
            #assert samples_.shape[0] == out_.shape[0]

        ## computation
        #cdef uint_t              i
        #cdef uint_t              j
        #cdef float_t             k
        #cdef float_t             n
        #cdef MixedBinomialSample s

        #for i in xrange(samples_.shape[0]):
            #k = 0.0
            #n = 0.0

            #for j in xrange(samples_.shape[1]):
                #s  = samples_[i, j]
                #k += s.k * weights_[i, j]
                #n += s.n * weights_[i, j]

            #out_[i] = (k + self._epsilon) / (n + self._epsilon)

        #return out_

    #@property
    #def parameter_dtype(self):
        #"""
        #Type of a distribution parameter.
        #"""

        #return self._parameter_dtype

    #@property
    #def sample_dtype(self):
        #"""
        #Type of a sample.
        #"""

        #return self._sample_dtype

#class LowMixedBinomial(object):
    #"""
    #Low-level operations of the binomial distribution.

    #Relevant types:
    #- parameter : f64
    #- sample    : {u32 n; u32 k;}
    #"""

    #def __init__(self, module):
        #"""
        #Initialize.
        #"""

        #from llvm.core import Type

        #i32_t = Type.int(32)
        #f64_t = Type.double()

        #self._par_t = f64_t
        #self._sam_t = Type.struct([i32_t, i32_t])

    #def rv(self, b, par_p, out_p, prng):
        #"""
        #Return samples from this distribution.
        #"""

        #raise NotImplementedError()

    #def ll(self, b, par_p, sam_p, out_p):
        #"""
        #Compute log probability under this distribution.
        #"""

        #from llvm.core import (
            #Type,
            #Constant,
            #)

        #b.store(Constant.real(Type.double(), 42.0), out_p)

    #def ml(self, sam_loop, weight_loop, out_p, prng):
        #"""
        #Return the estimated maximum-likelihood parameter.
        #"""

        #raise NotImplementedError()

    #@property
    #def parameter_type(self):
        #"""
        #LLVM type of the distribution parameter.
        #"""

        #return self._par_t

    #@property
    #def sample_type(self):
        #"""
        #LLVM type of the distribution sample.
        #"""

        #return self._sam_t

#class MixedBinomial(Distribution):
    #"""
    #Operate on multiple binomials with a single common probability parameter.
    #"""

    #_parameter_dtype = numpy.dtype(numpy.float64)
    #_sample_dtype    = numpy.dtype([("n", numpy.uint32), ("k", numpy.uint32)])

    #def __init__(self, epsilon = 1e-3):
        #"""
        #Initialize.
        #"""

        #Distribution.__init__(self, LowMixedBinomial)

