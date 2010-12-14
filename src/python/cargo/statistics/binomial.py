"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import qy

from llvm.core import Type
from qy        import (
    get_qy,
    Function,
    StridedArray,
    )

class Binomial(object):
    """
    Build low-level operations of the binomial distribution.

    Relevant types:
    - parameter : {float64 p; uint32 n;}
    - sample    : uint32
    """

    def __init__(self, estimation_n = None):
        """
        Initialize.
        """

        self._parameter_dtype = numpy.dtype([("p", numpy.float64), ("n", numpy.uint32)])
        self._sample_dtype    = numpy.dtype(numpy.int32)
        self._prior_dtype     = numpy.dtype([("alpha", float), ("beta", float)])
        self._estimation_n    = estimation_n # XXX MASSIVE HACK; needs to go away

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

    @property
    def prior_dtype(self):
        """
        Type of the prior parameter.
        """

        return self._prior_dtype

    @property
    def marginal_dtype(self):
        """
        Type of the marginal parameter.
        """

        return self._parameter_dtype

def binomial_log_pdf(k, p, n):
    """
    Compute the binomial PMF.
    """

    name = "binomial_log_pdf_ddd"

    if name in get_qy().module.global_variables:
        pdf = Function.get_named(name)
    else:
        @Function.define(float, [float, float, float])
        def binomial_log_pdf_ddd(k, p, n):
            from qy.math import ln_choose

            @qy.if_(k > n)
            def _():
                qy.return_(-numpy.inf)

            @qy.if_(p == 0.0)
            def _():
                qy.return_(qy.select(k == 0.0, 0.0, -numpy.inf))

            @qy.if_(p == 1.0)
            def _():
                qy.return_(qy.select(k == n, 0.0, -numpy.inf))

            qy.return_(ln_choose(n, k) + k * qy.log(p) + (n - k) * qy.log1p(-p))

    return binomial_log_pdf_ddd(k, p, n)

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

        binomial_log_pdf(
            sample.data.load(),
            parameter.data.gep(0, 0).load(),
            parameter.data.gep(0, 1).load(),
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
            prior_data = qy.stack_allocate(self._model.prior_dtype)

            qy.value_from_any(1.0).store(prior_data.gep(0, 0))
            qy.value_from_any(1.0).store(prior_data.gep(0, 1))

            self._map(
                StridedArray.from_raw(prior_data, (), ()),
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

        binomial_ml(samples.data, weights.data, out.data)

    def map(self, prior, samples, weights, out):
        """
        Emit computation of the estimated MAP parameter.
        """

        @Function.define(
            Type.void(),
            [prior.data.type_, samples.data.type_, weights.data.type_, out.data.type_],
            )
        def binomial_map(prior_data, samples_data, weights_data, out_data):
            self._map(
                prior.using(prior_data),
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

        binomial_map(prior.data, samples.data, weights.data, out.data)

    def _map(self, prior, samples, weights, out):
        """
        Emit computation of the estimated MAP parameter.
        """

        total_k = qy.stack_allocate(float, 0.0)
        total_w = qy.stack_allocate(float, 0.0)

        @qy.for_(samples.shape[0])
        def _(n):
            weight = weights.at(n).data.load()
            sample = samples.at(n).data.load().cast_to(float)

            (total_k.load() + sample * weight).store(total_k)
            (total_w.load() + weight * float(self._model._estimation_n)).store(total_w)

        alpha = prior.data.gep(0, 0).load()
        beta  = prior.data.gep(0, 1).load()

        numerator   = total_k.load() + alpha - 1.0
        denominator = total_w.load() + alpha + beta - 2.0

        (numerator / denominator).store(out.data.gep(0, 0))
        qy.value_from_any(self._model._estimation_n).store(out.data.gep(0, 1))

        qy.return_()

    def given(self, parameter, samples, out):
        """
        Return the conditional distribution.
        """

        parameter.data.gep(0, 0).load().store(out.data.gep(0, 0))
        parameter.data.gep(0, 1).load().store(out.data.gep(0, 1))

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

        self._parameter_dtype = numpy.dtype(float)
        self._sample_dtype    = numpy.dtype([("k", numpy.uint32), ("n", numpy.uint32)])
        self._prior_dtype     = numpy.dtype([("alpha", float), ("beta", float)])
        self._average_dtype   = numpy.dtype(float)

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

    @property
    def prior_dtype(self):
        """
        Type of the distribution prior.
        """

        return self._prior_dtype

    @property
    def average_dtype(self):
        """
        Type of the distribution prior.
        """

        return self._average_dtype

class MixedBinomialEmitter(object):
    """
    Build low-level operations of the binomial distribution.
    """

    def __init__(self, model):
        """
        Initialize.
        """

        self._model = model

    def ll(self, parameter, sample, out):
        """
        Compute log probability under this distribution.
        """

        @Function.define(
            Type.void(),
            [parameter.data.type_, sample.data.type_, out.type_],
            )
        def mixed_binomial_ll(parameter_data, sample_data, out_data):
            self._ll(
                parameter.using(parameter_data),
                sample.using(sample_data),
                out_data,
                )

            qy.return_()

        mixed_binomial_ll(parameter.data, sample.data, out)

    def _ll(self, parameter, sample, out):
        """
        Compute log probability under this distribution.
        """

        p = parameter.data.load()
        k = sample.data.gep(0, 0).load()
        n = sample.data.gep(0, 1).load()

        if get_qy().test_for_nan:
            qy.assert_(p >= 0.0, "invalid p = %s"           , p   )
            qy.assert_(p <= 1.0, "invalid p = %s"           , p   )
            qy.assert_(k >= 0  , "invalid k = %s"           , k   )
            qy.assert_(n >= 0  , "invalid n = %s"           , n   )
            qy.assert_(k <= n  , "invalid k = %s (> n = %s)", k, n)

        binomial_log_pdf(k, p, n).store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        @Function.define(
            Type.void(),
            [samples.data.type_, weights.data.type_, out.data.type_],
            )
        def mixed_binomial_ml(samples_data, weights_data, out_data):
            prior_data = qy.stack_allocate(self._model.prior_dtype)

            qy.value_from_any(1.0).store(prior_data.gep(0, 0))
            qy.value_from_any(1.0).store(prior_data.gep(0, 1))

            self._map(
                StridedArray.from_raw(prior_data, (), ()),
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

            qy.return_()

        mixed_binomial_ml(samples.data, weights.data, out.data)

    def map(self, prior, samples, weights, out):
        """
        Emit computation of the estimated MAP parameter.
        """

        @Function.define(
            Type.void(),
            [prior.data.type_, samples.data.type_, weights.data.type_, out.data.type_],
            )
        def mixed_binomial_ml(prior_data, samples_data, weights_data, out_data):
            self._map(
                prior.using(prior_data),
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

            qy.return_()

        mixed_binomial_ml(prior.data, samples.data, weights.data, out.data)

    def _map(self, prior, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        total_k = qy.stack_allocate(float, 0.0)
        total_n = qy.stack_allocate(float, 0.0)

        @qy.for_(samples.shape[0])
        def _(n):
            weight   = weights.at(n).data.load()
            sample_k = samples.at(n).data.gep(0, 0).load().cast_to(float)
            sample_n = samples.at(n).data.gep(0, 1).load().cast_to(float)

            (total_k.load() + sample_k * weight).store(total_k)
            (total_n.load() + sample_n * weight).store(total_n)

        alpha = prior.data.gep(0, 0).load()
        beta  = prior.data.gep(0, 1).load()

        numerator   = total_k.load() + alpha - 1.0
        denominator = total_n.load() + alpha + beta - 2.0

        (numerator / denominator).store(out.data)

    def given(self, parameter, samples, out):
        """
        Return the conditional distribution.
        """

        parameter.data.load().store(out.data)

    def average(self, weights, parameters, out):
        """
        Emit computation of the average parameter.
        """

        @Function.define(
            Type.void(),
            [weights.data.type_, parameters.data.type_, out.data.type_],
            )
        def binomial_average(weights_data, parameters_data, out_data):
            self._average(
                weights.using(weights_data),
                parameters.using(parameters_data),
                out.using(out_data),
                )

        binomial_average(weights.data, parameters.data, out.data)

    def _average(self, weights, parameters, out):
        """
        Emit computation of the average parameter.
        """

        qy.value_from_any(0.0).store(out.data)

        @qy.for_(parameters.shape[0])
        def _(n):
            weight    = weights.at(n).data.load()
            parameter = parameters.at(n).data.load()

            (out.data.load() + weight * parameter).store(out.data)

        qy.return_()

