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
    broadcast,
    int32_t,
    int64_t,
    )

log = get_logger(__name__)

cdef void zorro(int32_t i, int32_t j):
    print i, j

def get_zorro():
    from llvm.ee   import TargetData
    from llvm.core import (
        Type,
        Constant,
        )

    uintptr_t = Type.int(TargetData.new("target").pointer_size * 8)
    zorro_t   = Type.pointer(Type.function(Type.void(), [Type.int(32), Type.int(32)]))

    return Constant.int(uintptr_t, <long>&zorro).inttoptr(zorro_t)

#class FiniteMixture(object):
    #"""
    #An arbitrary finite homogeneous mixture distribution.
    #"""

    #def __init__(self, distribution, K, iterations = 256, convergence = 1e-8):
        #"""
        #Initialize.
        #"""

        #self._distribution   = distribution
        #self._K              = K
        #self._iterations     = iterations
        #self._convergence    = convergence
        #self._parameter_type = \
            #Type.array(
                #Type.struct([Type.double(), distribution.parameter_type]),
                #K,
                #)

    #def rv(self, parameters, out, random = numpy.random):
        #"""
        #Make a draw from this mixture distribution.
        #"""

        ## identify the common prefix
        #if self._distribution.sample_dtype.shape:
            #out_prefix = out.shape[:-len(self._distribution.sample_dtype.shape)]
        #else:
            #out_prefix = out.shape

        #selected = numpy.empty(out_prefix, dtype = self._distribution.parameter_dtype)

        ## select the relevant components
        #extension  = (1,) * (len(selected.shape) - len(parameters.shape)) + parameters.shape
        #components = \
            #numpy.reshape(
                #parameters["c"],
                #extension + self._distribution.parameter_dtype.shape,
                #)
        #mixing     = numpy.reshape(parameters["p"], extension)

        #less   = 1 + len(self._distribution.parameter_dtype.shape)
        #re_max = tuple([s - 1 for s in components.shape[:-less]])

        #for i in numpy.ndindex(selected.shape):
            #re_i = tuple(map(min, re_max, i))
            #j    = numpy.nonzero(random.multinomial(1, mixing[re_i]))

            #selected[i] = components[re_i + j]

        ## generate random variates
        #return self._distribution.rv(selected, out, random)

    ##def ll(self, block, parameter, sample):
        ##"""
        ##Compute finite-mixture log-likelihood.
        ##"""

        ### computation
        ##from cargo.statistics.base import log_add

        ##ll_out  = self._distribution.ll(parameters["c"], samples)
        ##ll_out += numpy.log(parameters["p"])
        ##pre_out = log_add.reduce(ll_out, -1)

        ##numpy.sum(pre_out, -1, out = out)

        ##return out

    #def ml(
                                   #self,
        #ndarray                    samples, # ndim = 2
        #ndarray[float_t, ndim = 2] weights,
        #ndarray                    out,     # ndim = 1
                                   #random = numpy.random,
        #):
        #"""
        #Use EM to estimate mixture parameters.
        #"""

        ## arguments
        #assert samples.shape[0] == weights.shape[0]
        #assert samples.shape[1] == weights.shape[1]

        #if not numpy.all(weights == 1.0):
            #raise NotImplementedError("non-unit sample weighting not yet supported")

        #if out is None:
            #out = numpy.empty(samples.shape[0], self._parameter_dtype)
        #else:
            #assert samples.shape[0] == out.shape[0]

        ## computation
        #log.detail("estimating finite mixture from %i samples" % samples.shape[1])

        #for i in xrange(samples.shape[0]):
            #out[i] = self._ml(samples[i], weights[i], random)

        #return out

    #def _ml(
                                   #self,
        #ndarray                    samples_N,
        #ndarray[float_t, ndim = 1] weights_N,
                                   #random = numpy.random,
        #):
        #"""
        #Use EM to estimate mixture parameters.
        #"""

        ## mise en place
        #cdef size_t N = samples_N.shape[0]
        #cdef size_t K = self._K

        #d = self._distribution
        #p = numpy.empty((), self._parameter_dtype)

        ## generate a random initial state
        #seeds = random.randint(N, size = K)

        #d.ml(samples_N[seeds][:, None], weights_N[seeds][:, None], p["c"], random)

        #p["p"]  = random.rand(K)
        #p["p"] /= numpy.sum(p["p"])

        ## run EM until convergence
        #last_r_KN = None
        #r_KN      = numpy.empty((K, N))

        #for i in xrange(self._iterations):
            ## evaluate responsibilities
            #d.ll(p["c"][:, None], samples_N, r_KN)

            #numpy.exp(r_KN, r_KN)

            #r_KN *= p["p"][:, None]
            #r_KN /= numpy.sum(r_KN, 0)

            ## make maximum-likelihood estimates
            #d.ml(samples_N, r_KN, p["c"], random)

            #p["p"] = numpy.sum(r_KN, 1) / N

            ## check for convergence
            #if last_r_KN is None:
                #last_r_KN = numpy.empty((K, N))
            #else:
                #difference = numpy.sum(numpy.abs(r_KN - last_r_KN))

                #log.detail(
                    #"iteration %i < %i ; delta %e >? %e",
                    #i,
                    #self._iterations,
                    #difference,
                    #self._convergence,
                    #)

                #if difference < self._convergence:
                    #break

            #(last_r_KN, r_KN) = (r_KN, last_r_KN)

        ## done
        #return p

    #def given(self, parameters, samples, out = None):
        #"""
        #Return the conditional distribution.
        #"""

        ## arguments
        #from cargo.numpy import semicast

        #parameters = numpy.asarray(parameters, self._parameter_dtype.base)
        #samples    = numpy.asarray(samples   , self.sample_dtype         )

        #if out is None:
            #(parameters, samples) = \
                #semicast(
                    #(parameters, -1                                   ),
                    #(samples   , -len(self.sample_dtype.shape) or None),
                    #)

            #print parameters.shape, samples.shape

            #out = numpy.empty_like(parameters)
        #else:
            #(parameters, samples, _) = \
                #semicast(
                    #(parameters, -1                                   ),
                    #(samples   , -len(self.sample_dtype.shape) or None),
                    #(out       , -1                                   ),
                    #)

            #assert out.shape == parameters.shape

        ## compute posterior mixture parameters
        #out["p"]  = parameters["p"]

        #ll = self._distribution.ll(parameters["c"], samples[..., None])

        #if ll.ndim > 1:
            #sum_ll = numpy.sum(ll, -2)
        #else:
            #sum_ll = ll

        #out["p"] *= numpy.exp(sum_ll)
        #out["p"] /= numpy.sum(out["p"], -1)[..., None]

        ## compute posterior mixture components
        #self._distribution.given(parameters["c"], samples[..., None], out["c"])

        ## done
        #return out

    #@property
    #def distribution(self):
        #"""
        #Return the mixture components.
        #"""

        #return self._distribution

    #@property
    #def parameter_type(self):
        #"""
        #Return the parameter type.
        #"""

        #return self._parameter_type

    #@property
    #def sample_type(self):
        #"""
        #Return the sample type.
        #"""

        #return self._distribution.sample_type

class RestartingML(object):
    """
    Wrap a distribution with a restarting ML estimator.
    """

    def __init__(self, distribution, restarts = 8):
        """
        Initialize.
        """

        self._distribution = distribution
        self._restarts     = restarts

    def rv(self, parameters, out, random = numpy.random):
        """
        Make a draw from this mixture distribution.
        """

        return self._distribution.rv(parameters, out, random)

    def ll(self, parameters, samples, out = None):
        """
        Compute finite-mixture log-likelihood.
        """

        return self._distribution.ll(parameters, samples, out)

    def ml(self, samples, weights, out, random = numpy.random):
        """
        Use EM to estimate mixture parameters.
        """

        raise NotImplementedError()

    def given(self, parameters, samples, out = None):
        """
        Return the conditional distribution.
        """

        return self._distribution.given(parameters, samples, out)

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

        return self._distribution.parameter_dtype

    @property
    def sample_dtype(self):
        """
        Return the sample type.
        """

        return self._distribution.sample_dtype

