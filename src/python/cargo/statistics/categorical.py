"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import qy

from llvm.core import Type
from qy        import (
    get_qy,
    Function,
    )

class Categorical(object):
    """
    The categorical (or "discrete") distribution.
    """

    def __init__(self, estimation_n = None, epsilon = 0.0):
        """
        Initialize.
        """

        self._parameter_dtype = numpy.dtype([("p", numpy.float64), ("n", numpy.uint32)])
        self._sample_dtype    = numpy.dtype(numpy.int32)
        self._estimation_n    = estimation_n # XXX MASSIVE HACK; needs to go away
        self._epsilon         = epsilon

    def get_emitter(self):
        """
        Return IR emitter.
        """

        return BinomialEmitter(self)

    @property
    def parameter_dtype(self):
        """
        Type of the distribution parameter.
        """

        return self._parameter_dtype

    @property
    def sample_dtype(self):
        """
        Type of the distribution sample.
        """

        return self._sample_dtype

def binomial_pdf(k, p, n):
    """
    Compute the binomial PDF function.
    """

    name = "gsl_ran_binomial_pdf"

    if name in get_qy().module.global_variables:
        pdf = Function.get_named(name)
    else:
        import llvm.core

        from ctypes import c_uint

        pdf = Function.named(name, float, [c_uint, float, c_uint])

        pdf._value.add_attribute(llvm.core.ATTR_READONLY)
        pdf._value.add_attribute(llvm.core.ATTR_NO_UNWIND)

    return pdf(k, p, n)

class BinomialEmitter(object):
    """
    Build low-level operations of the binomial distribution.
    """

    def __init__(self, model):
        """
        Initialize.
        """

        # members
        self._model = model

        # link the GSL
        import ctypes

        from ctypes      import CDLL
        from ctypes.util import find_library

        CDLL(find_library("cblas"), ctypes.RTLD_GLOBAL)
        CDLL(find_library("gsl"  ), ctypes.RTLD_GLOBAL)

    def ll(self, parameter, sample, out):
        """
        Compute log probability under this distribution.
        """

        @Function.define(
            Type.void(),
            [parameter.data.type_, sample.data.type_, out.type_],
            )
        def binomial_ll(parameter_data, sample_data, out_data):
            self._ll(
                parameter.using(parameter_data),
                sample.using(sample_data),
                out_data,
                )

            qy.return_()

        binomial_ll(parameter.data, sample.data, out)

    def _ll(self, parameter, sample, out):
        """
        Compute log probability under this distribution.
        """

        qy.log(
            binomial_pdf(
                sample.data.load(),
                parameter.data.gep(0, 0).load(),
                parameter.data.gep(0, 1).load(),
                ),
            ) \
            .store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        @Function.define(
            Type.void(),
            [samples.data.type_, weights.data.type_, out.data.type_],
            )
        def binomial_ml(samples_data, weights_data, out_data):
            self._ml(
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

            qy.return_()

        binomial_ml(samples.data, weights.data, out.data)

    def _ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        total_k = qy.stack_allocate(float, 0.0)
        total_w = qy.stack_allocate(float, 0.0)

        @qy.for_(samples.shape[0])
        def _(n):
            weight = weights.at(n).data.load()
            sample = samples.at(n).data.load().cast_to(float)

            (total_k.load() + sample * weight).store(total_k)
            (total_w.load() + weight * float(self._model._estimation_n)).store(total_w)

        final_ratio = \
              (total_k.load() + self._model._epsilon) \
            / (total_w.load() + self._model._epsilon)

        final_ratio.store(out.data.gep(0, 0))
        qy.value_from_any(self._model._estimation_n).store(out.data.gep(0, 1))

#class Multinomial(Distribution):
    #"""
    #The multinomial distribution.

    #Relevant types:
    #- sample: D-shaped uint ndarray
    #- sequence: ND-shaped uint ndarray
    #"""

    #def __init__(self, beta, norm = 1):
        #"""
        #Instantiate the distribution.

        #@param beta: The distribution parameter vector.
        #"""

        ## initialization
        #self._beta = numpy.asarray(beta)
        #self._norm = norm

    #def random_variate(self, random = numpy.random):
        #"""
        #Return a sample from this distribution.
        #"""

        #return random.multinomial(self._norm, self._beta).astype(numpy.uint)

    #def random_variates(self, size, random = numpy.random):
        #"""
        #Return an array of samples from this distribution.
        #"""

        #return random.multinomial(self._norm, self._beta, size).astype(numpy.uint)

    #def log_likelihood(self, sample):
        #"""
        #Return the log probability of C{sample} under this distribution.
        #"""

        ## mise en place
        #cdef numpy.ndarray[double]       beta_D   = self._beta
        #cdef numpy.ndarray[numpy.uint_t] counts_D = sample

        #assert counts_D.shape[0] == beta_D.shape[0]

        ## calculate
        #cdef double        lp = 0.0
        #cdef unsigned long n  = 0

        #for d in xrange(counts_D.shape[0]):
            #n += counts_D[d]

            #if beta_D[d] > 0.0:
                #lp += log(beta_D[d]) * counts_D[d] - ln_gamma(counts_D[d] + 1)
            #else:
                #lp += -DBL_MAX * counts_D[d] - ln_gamma(counts_D[d] + 1)

        #return lp + ln_gamma(n + 1)

    #@cython.boundscheck(False)
    #def add_log_likelihoods(self, samples, to):
        #"""
        #Add the log likelihoods of C{samples} under this distribution.
        #"""

        ## mise en place
        #cdef numpy.ndarray[double]                 beta_D     = self._beta
        #cdef numpy.ndarray[numpy.uint_t, ndim = 2] samples_ND = numpy.asarray(samples, numpy.uint)
        #cdef numpy.ndarray[double]                 to_N       = to

        #assert samples_ND.shape[0] == to_N.shape[0]
        #assert samples_ND.shape[1] == beta_D.shape[0]

        ## calculate
        #cdef size_t        n
        #cdef size_t        d
        #cdef unsigned long m
        #cdef double        p

        #for n in xrange(samples_ND.shape[0]):
            #m = 0
            #p = 0.0

            #for d in xrange(samples_ND.shape[1]):
                #m += samples_ND[n, d]

                #if beta_D[d] > 0.0:
                    #p += log(beta_D[d]) * samples_ND[n, d] - ln_gamma(samples_ND[n, d] + 1)
                #else:
                    #p += -DBL_MAX * samples_ND[n, d] - ln_gamma(samples_ND[n, d] + 1)

            #to_N[n] += p + ln_gamma(m + 1)

    #def total_log_likelihood(self, samples):
        #"""
        #Return the log likelihood of C{samples} under this distribution.
        #"""

        #return self.log_likelihood(numpy.sum(samples, 0))

    #@property
    #def beta(self):
        #"""
        #Return the multinomial parameter vector.
        #"""

        #return self._beta

    #@property
    #def log_beta(self):
        #"""
        #Return the multinomial log parameter vector.
        #"""

        #return numpy.nan_to_num(numpy.log(self._beta))

#class MultinomialEstimator(Estimator):
    #"""
    #Estimate the parameters of a multinomial distribution.
    #"""

    #def __init__(self, norm = 1, epsilon = 1e-3):
        #"""
        #Initialize.
        #"""

        #self._norm = norm

        #if epsilon is None:
            #self._epsilon = 0.0
        #else:
            #self._epsilon = epsilon

    #@cython.boundscheck(False)
    #def estimate(self, samples, random = numpy.random, weights = None):
        #"""
        #Return the estimated maximum likelihood distribution.
        #"""

        ## parameters
        #cdef numpy.ndarray[uint_t, ndim = 2] samples_ND = numpy.asarray(samples, numpy.uint)
        #cdef numpy.ndarray[double]           weights_N

        #if weights is None:
            #weights_N = numpy.ones(samples_ND.shape[0])
        #else:
            #weights_N = numpy.asarray(weights)

        #assert samples_ND.shape[0] == weights_N.shape[0]

        ## estimate
        #cdef size_t                D      = samples_ND.shape[1]
        #cdef numpy.ndarray[double] beta_D = numpy.zeros(D)
        #cdef size_t                n
        #cdef size_t                d

        #for n in xrange(samples_ND.shape[0]):
            #for d in xrange(D):
                #beta_D[d] += weights_N[n] * samples_ND[n, d]

        ## normalization and heuristic smoothing
        #cdef double epsilon  = self._epsilon
        #cdef double sum_beta = 0.0

        #for d in xrange(D):
            #beta_D[d] += epsilon
            #sum_beta  += beta_D[d]

        #for d in xrange(D):
            #beta_D[d] /= sum_beta

        #return Multinomial(beta_D, self._norm)

