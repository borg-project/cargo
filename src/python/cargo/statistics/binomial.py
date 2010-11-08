"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from llvm.core import (
    Type,
    Builder,
    Constant,
    )
from cargo.llvm.high_level import (
    high,
    HighFunction,
    )

class Binomial(object):
    """
    Build low-level operations of the binomial distribution.

    Relevant types:
    - parameter : {float64 p; uint32 n;}
    - sample    : uint32
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

        from ctypes import c_uint

        pdf = HighFunction.named("gsl_ran_binomial_pdf", float, [c_uint, float, c_uint])

        high.log(
            pdf(
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

        @HighFunction.define(
            Type.void(),
            [samples.data.type_, weights.data.type_, out.data.type_],
            )
        def binomial_ml(samples_data, weights_data, out_data):
            self._ml(
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

            high.return_()

        binomial_ml(samples.data, weights.data, out.data)

    def _ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        total_k = high.stack_allocate(float, 0.0)
        total_w = high.stack_allocate(float, 0.0)

        @high.for_(samples.shape[0])
        def _(n):
            weight = weights.at(n).data.load()
            sample = samples.at(n).data.load().cast_to(float)

            (total_k.load() + sample * weight).store(total_k)
            (total_w.load() + weight * float(self._model._estimation_n)).store(total_w)

        final_ratio = \
              (total_k.load() + self._model._epsilon) \
            / (total_w.load() + self._model._epsilon)

        final_ratio.store(out.data.gep(0, 0))
        high.value_from_any(self._model._estimation_n).store(out.data.gep(0, 1))

class MixedBinomial(object):
    """
    The "mixed binomial" distribution.

    Relevant types:
    - parameter : float64 p
    - sample    : {uint32 k; uint32 n;}
    """

    def __init__(self):
        """
        Initialize.
        """

        self._parameter_dtype = numpy.dtype(numpy.float64)
        self._sample_dtype    = numpy.dtype([("k", numpy.uint32), ("n", numpy.uint32)])

    def get_emitter(self):
        """
        Return IR emitter.
        """

        return MixedBinomialEmitter(self)

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

class MixedBinomialEmitter(object):
    """
    Build low-level operations of the binomial distribution.
    """

    def __init__(self, model):
        """
        Initialize.
        """

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

        from ctypes import c_uint

        pdf = HighFunction.named("gsl_ran_binomial_pdf", float, [c_uint, float, c_uint])

        p = parameter.data.load()
        k = sample.data.gep(0, 0).load()
        n = sample.data.gep(0, 1).load()

        if high.test_for_nan:
            high.assert_(p >= 0.0, "invalid p = %s"           , p   )
            high.assert_(p <= 1.0, "invalid p = %s"           , p   )
            high.assert_(k >= 0  , "invalid k = %s"           , k   )
            high.assert_(n >= 0  , "invalid n = %s"           , n   )
            high.assert_(k <= n  , "invalid k = %s (> n = %s)", k, n)

        high.log(pdf(k, p, n)).store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        @HighFunction.define(
            Type.void(),
            [samples.data.type_, weights.data.type_, out.data.type_],
            )
        def mixed_binomial_ml(samples_data, weights_data, out_data):
            self._ml(
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

            high.return_()

        mixed_binomial_ml(samples.data, weights.data, out.data)

    def _ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        total_k = high.stack_allocate(float, 0.0)
        total_n = high.stack_allocate(float, 0.0)

        @high.for_(samples.shape[0])
        def _(n):
            weight   = weights.at(n).data.load()
            sample_k = samples.at(n).data.gep(0, 0).load().cast_to(float)
            sample_n = samples.at(n).data.gep(0, 1).load().cast_to(float)

            if high.test_for_nan:
                high.assert_(weight   >= 0.0     , "invalid weight = %s"      , weight            )
                high.assert_(sample_k >= 0       , "invalid k = %s"           , sample_k          )
                high.assert_(sample_n >= 0       , "invalid n = %s"           , sample_n          )
                high.assert_(sample_k <= sample_n, "invalid k = %s (> n = %s)", sample_k, sample_n)

            (total_k.load() + sample_k * weight).store(total_k)
            (total_n.load() + sample_n * weight).store(total_n)

        epsilon = numpy.finfo(float).eps
        ratio   = ((total_k.load() + epsilon) / (total_n.load() + epsilon)) / (1.0 / (1.0 - 2 * epsilon)) + epsilon

        ratio.store(out.data)

    def given(self, parameter, samples, out):
        """
        Return the conditional distribution.
        """

        parameter.data.load().store(out.data)

