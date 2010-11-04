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

    def get_emitter(self, module):
        """
        Return IR emitter.
        """

        return BinomialEmitter(self, module)

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

    def __init__(self, model, module):
        """
        Initialize.
        """

        # members
        self._model  = model
        self._module = module

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

        from cargo.llvm import this_builder

        compute = \
            HighFunction.new_named(
                "binomial_ml",
                Type.void(),
                [samples.data.type_, weights.data.type_, out.data.type_],
                )
        entry = compute.low.append_basic_block("entry")

        with this_builder(Builder.new(entry)) as builder:
            self._ml(
                samples.using(compute.argument_values[0]),
                weights.using(compute.argument_values[1]),
                out.using(compute.argument_values[2]),
                )

            builder.ret_void()

        compute(samples.data, weights.data, out.data)

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

    def __init__(self, epsilon = 1e-3):
        """
        Initialize.
        """

        self._epsilon         = epsilon
        self._parameter_dtype = numpy.dtype(numpy.float64)
        self._sample_dtype    = numpy.dtype([("k", numpy.uint32), ("n", numpy.uint32)])

    def get_emitter(self, module):
        """
        Return IR emitter.
        """

        return MixedBinomialEmitter(self, module)

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

    def __init__(self, model, module):
        """
        Initialize.
        """

        self._model  = model
        self._module = module

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

        from cargo.llvm import this_builder

        compute = \
            HighFunction.new_named(
                "mixed_binomial_ml",
                Type.void(),
                [samples.data.type_, weights.data.type_, out.data.type_],
                )
        entry = compute.low.append_basic_block("entry")

        with this_builder(Builder.new(entry)) as builder:
            self._ml(
                samples.using(compute.argument_values[0]),
                weights.using(compute.argument_values[1]),
                out.using(compute.argument_values[2]),
                )

            builder.ret_void()

        compute(samples.data, weights.data, out.data)

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

            #high.printf(
                #"sample %i: k = %i; n = %i; w = %.2f",
                #n,
                #sample_k,
                #sample_n,
                #weight,
                #)

            if high.test_for_nan:
                high.assert_(weight   >= 0.0     , "invalid weight = %s"      , weight            )
                high.assert_(sample_k >= 0       , "invalid k = %s"           , sample_k          )
                high.assert_(sample_n >= 0       , "invalid n = %s"           , sample_n          )
                high.assert_(sample_k <= sample_n, "invalid k = %s (> n = %s)", sample_k, sample_n)

            (total_k.load() + sample_k * weight).store(total_k)
            (total_n.load() + sample_n * weight).store(total_n)

        numerator   = (total_k.load() + self._model._epsilon)
        denominator = (total_n.load() + self._model._epsilon)
        final_ratio = numerator / denominator

        if high.test_for_nan:
            high.assert_(
                ~final_ratio.is_nan,
                "ratio (%s / %s) is not a number",
                numerator,
                denominator,
                )

        final_ratio.store(out.data)

