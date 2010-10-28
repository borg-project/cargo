"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.llvm import (
    high,
    StridedArray,
    )

class Tuple(object):
    """
    A tuple of independent distributions.
    """

    def __init__(self, distributions):
        """
        Initialize.
        """

        # sanitize arguments
        def normalize(distribution):
            """
            The canonical distribution argument is (model, count).
            """

            if isinstance(distribution, tuple):
                return distribution
            else:
                return (distribution, 1)

        self._distributions = map(normalize, distributions)

        # parameter and sample types
        parameter_fields = []
        sample_fields    = []

        for (i, (distribution, count)) in enumerate(self._distributions):
            field_name        = "d%i" % i
            parameter_fields += [(field_name, distribution.parameter_dtype, (count,))]
            sample_fields    += [(field_name, distribution.sample_dtype   , (count,))]

        self._parameter_dtype = numpy.dtype(parameter_fields)
        self._sample_dtype    = numpy.dtype(sample_fields)

    def get_emitter(self, module):
        """
        Return an IR emitter.
        """

        return TupleEmitter(self, module)

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

class TupleEmitter(object):
    """
    Emitter for the tuple distribution.
    """

    def __init__(self, model, module):
        """
        Initialize.
        """

        self._model    = model
        self._module   = module
        self._emitters = [d.get_emitter(module) for (d, _) in model._distributions]

    def ll(self, parameter, sample, out):
        """
        Compute log likelihood under this distribution.
        """

        high.value(0.0).store(out)

        for (i, (_, count)) in enumerate(self._model._distributions):
            @high.for_(count)
            def _(j):
                previous_total = out.load()

                self._emitters[i].ll(
                    StridedArray.from_typed_pointer(parameter.data.gep(0, i, j)),
                    StridedArray.from_typed_pointer(sample.data.gep(0, i, j)),
                    out,
                    )

                (out.load() + previous_total).store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        for (i, (_, count)) in enumerate(self._model._distributions):
            @high.for_(count)
            def _(j):
                self._emitters[i].ml(
                    samples.extract(0, i, j),
                    weights,
                    StridedArray.from_typed_pointer(out.data.gep(0, i, j)),
                    )

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

