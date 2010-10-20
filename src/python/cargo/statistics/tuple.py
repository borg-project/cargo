"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

#class TupleIR(object):
    #"""
    #Low-level operations of the tuple distribution.
    #"""

    #def __init__(self, module, distributions):
        #"""
        #Initialize.
        #"""

        #self._distributions  = distributions
        #self._parameter_type = Type.packed_struct([d.parameter_type for d in distributions])
        #self._sample_type    = Type.packed_struct([d.sample_type for d in distributions])

    #def ll(self, b, parameter, sample):
        #"""
        #Compute log probability under this distribution.
        #"""

        #from llvm.core import (
            #Type,
            #Constant,
            #)

        #o = Constant.real(Type.double(), 0.0)

        #for (i, d) in enumerate(self._distributions):
            #l = d.ll(
                    #b,
                    #b.extract_element(parameter, [i]),
                    #b.extract_element(sample   , [i]),
                    #)
            #o = b.add(o, l)

        #return o

    #@property
    #def distributions(self):
        #"""
        #Return the inner distributions.
        #"""

        #return self._distributions

    #@property
    #def parameter_type(self):
        #"""
        #Type of distribution parameter(s).
        #"""

        #return self._parameter_type

    #@property
    #def sample_type(self):
        #"""
        #Type of distribution parameter(s).
        #"""

        #return self._sample_type

#class Tuple(object):
    #"""
    #A tuple of independent distributions.
    #"""

    #def __init__(self, distributions):
        #"""
        #Initialize.
        #"""

        #self._distributions   = distributions
        #self._names           = ["d%i" % i for i in xrange(len(distributions))]
        #self._parameter_dtype = \
            #numpy.dtype([
                #(n, d.parameter_dtype)
                #for n in self._names
                #for d in distributions
                #])
        #self._sample_dtype    = \
            #numpy.dtype([
                #(n, d.sample_dtype)
                #for n in self._names
                #for d in distributions
                #])

    #def rv(
                          #self,
        #ndarray           parameters, # ndim = 1
        #ndarray           out,        # ndim = 2
                          #random = numpy.random,
        #):
        #"""
        #Return samples from this distribution.
        #"""

        ### arguments
        ##assert parameters.shape[0] == out.shape[0]

        ### computation
        ##cdef size_t            i
        ##cdef size_t            j
        ##cdef BinomialParameter p

        ##for i in xrange(out.shape[0]):
            ##p         = parameters[i]
            ##out[i, :] = random.binomial(p.n, p.p, out.shape[1])

        #return out

    #def ll(
                          #self,
        #ndarray           parameters, # ndim = 1
        #ndarray           samples,    # ndim = 2
        #ndarray           out = None, # ndim = 2
        #):
        #"""
        #Return the log probability of samples under this distribution.
        #"""

        ## arguments
        #assert len(parameters.dtype.names) == len(self._distributions)
        #assert samples.shape[0] == parameters.shape[0]

        #if out is None:
            #out = numpy.empty((samples.shape[0], samples.shape[1]), numpy.float_)
        #else:
            #assert samples.shape[0] == out.shape[0]
            #assert samples.shape[1] == out.shape[1]

        ## computation
        #for (name, distribution) in zip(self._names, self._distributions):
            #distribution.ll(parameters[name], samples[name], out[name])

        #return out

    #def ml(
                                   #self,
        #ndarray                    samples, # ndim = 1
        #ndarray[float_t, ndim = 1] weights,
                                   #random = numpy.random,
        #):
        #"""
        #Return the estimated maximum-likelihood parameter.
        #"""

        ##parameters = numpy.

    #@property
    #def distributions(self):
        #"""
        #Return the inner distributions.
        #"""

        #return self._distributions

    #@property
    #def parameter_dtype(self):
        #"""
        #Type of distribution parameter(s).
        #"""

        #return self._parameter_dtype

    #@property
    #def sample_dtype(self):
        #"""
        #Type of distribution parameter(s).
        #"""

        #return self._sample_dtype

#class Tuple(Distribution):
    #"""
    #A tuple of independent distributions.
    #"""

    #def __init__(self, distributions):
        #"""
        #Initialize.
        #"""

        ## members
        #self._distributions   = distributions
        #self._names           = ["d%i" % i for i in xrange(len(distributions))]
        ##self._parameter_dtype = \
            ##numpy.dtype([
                ##(n, d.parameter_dtype)
                ##for n in self._names
                ##for d in distributions
                ##])
        ##self._sample_dtype    = \
            ##numpy.dtype([
                ##(n, d.sample_dtype)
                ##for n in self._names
                ##for d in distributions
                ##])

        ## base
        #Distribution.__init__(self)

    #def core(self, module):
        #"""
        #Build LLVM operations for this distribution.
        #"""

        #LowTuple.__init__(self, module, [d.core(module) for d in distributions])

