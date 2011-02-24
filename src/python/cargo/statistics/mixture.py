"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys
import numpy
import llvm.core
import qy

from qy         import (
    get_qy,
    Function,
    Variable,
    StridedArray,
    StridedArrays,
    )
from llvm.core  import (
    Type,
    Constant,
    )
from cargo.log  import get_logger

logger = get_logger(__name__)

def log_add_double(x, y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic.
        Kingsbury and Rayner, 1970.
    """

    if "log_add_d" in get_qy().module.global_variables:
        log_add_d = Function.get_named("log_add_d")
    else:
        @Function.define(float, [float, float])
        def log_add_d(x_in, y_in):
            s = x_in >= y_in
            a = qy.select(s, x_in, y_in)

            @qy.if_else(a == -numpy.inf)
            def _(then):
                if then:
                    qy.return_(-numpy.inf)
                else:
                    qy.return_(a + qy.log1p(qy.exp(qy.select(s, y_in, x_in) - a)))

    return log_add_d(x, y)

class FiniteMixture(object):
    """
    An arbitrary finite homogeneous mixture distribution.
    """

    def __init__(self, distribution, K, iterations = 256, convergence = 1e-8):
        """
        Initialize.
        """

        self._distribution    = distribution
        self._K               = K
        self._iterations      = iterations
        self._convergence     = convergence
        self._parameter_dtype = \
            numpy.dtype((
                [
                    ("p", numpy.float64),
                    ("c", distribution.parameter_dtype),
                    ],
                (K,),
                ))
        self._prior_dtype = numpy.dtype((distribution.prior_dtype, (K,)))

    def get_emitter(self):
        """
        Return an IR emitter for this distribution.
        """

        return FiniteMixtureEmitter(self)

    def posterior(self, parameter, samples):
        """
        Return the posterior mixture weights.
        """

        # compute the component likelihoods
        post = numpy.ndarray(self.K)

        for i in xrange(self.K):
            ll = parameter[i]["p"]

            for j in xrange(len(samples)):
                ll += self.distribution.ll(parameter[i]["c"], samples[j])

            post[i] = ll

        # normalize and exponentiate
        from cargo.statistics.functions import log_plus_all

        post[:] -= log_plus_all(post)

        numpy.exp(post, post)

        return post

    @property
    def parameter_dtype(self):
        """
        Return the parameter type.
        """

        return self._parameter_dtype

    @property
    def sample_dtype(self):
        """
        Return the sample type.
        """

        return self._distribution.sample_dtype

    @property
    def prior_dtype(self):
        """
        Return the prior type.
        """

        return self._prior_dtype

    @property
    def marginal_dtype(self):
        """
        Return the marginal dtype.
        """

        return self._distribution.average_dtype

    @property
    def K(self):
        """
        The number of mixture components.
        """

        return self._K

    @property
    def distribution(self):
        """
        Return the mixture components.
        """

        return self._distribution

class FiniteMixtureEmitter(object):
    """
    Emit IR for the FiniteMixture distribution.
    """

    def __init__(self, model):
        """
        Initialize.
        """

        self._model       = model
        self._sub_emitter = self._model.distribution.get_emitter()

    def ll(self, parameter, sample, out):
        """
        Compute finite-mixture log-likelihood.
        """

        @Function.define(
            Type.void(),
            [parameter.data.type_, sample.data.type_, out.type_],
            )
        def finite_mixture_ll(parameter_data, sample_data, out_data):
            self._ll(
                parameter.using(parameter_data),
                sample.using(sample_data),
                out_data,
                )

            qy.return_()

        finite_mixture_ll(parameter.data, sample.data, out)

    def _ll(self, parameter, sample, out):
        """
        Compute finite-mixture log-likelihood.
        """

        total        = qy.stack_allocate(float, -numpy.inf, "total")
        component_ll = qy.stack_allocate(float)

        @qy.for_(self._model._K)
        def _(index):
            component = parameter.at(index)

            self._sub_emitter.ll(
                StridedArray.from_typed_pointer(component.data.gep(0, 1)),
                sample,
                component_ll,
                )

            log_add_double(
                total.load(),
                qy.log(component.data.gep(0, 0).load()) + component_ll.load(),
                ) \
                .store(total)

        total.load().store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        @Function.define(
            Type.void(),
            [samples.data.type_, weights.data.type_, out.data.type_],
            )
        def finite_mixture_ml(samples_data, weights_data, out_data):
            self._ml(
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                )

        finite_mixture_ml(samples.data, weights.data, out.data)

    # XXX def _ml

    def map(self, prior, samples, weights, out, initializations = 16):
        """
        Emit computation of the estimated MAP parameter.
        """

        @Function.define(
            Type.void(),
            [prior.data.type_, samples.data.type_, weights.data.type_, out.data.type_],
            )
        def finite_mixture_map(prior_data, samples_data, weights_data, out_data):
            self._map(
                prior.using(prior_data),
                samples.using(samples_data),
                weights.using(weights_data),
                out.using(out_data),
                initializations,
                )

        finite_mixture_map(prior.data, samples.data, weights.data, out.data)

    def _map_initialize(self, prior, samples, weights, out, initializations):
        """
        Emit parameter initialization for EM.
        """

        # generate a random initial component assignment
        K = self._model._K
        N = samples.shape[0]

        total   = qy.stack_allocate(float)
        best_ll = qy.stack_allocate(float, -numpy.inf)

        assigns      = StridedArray.heap_allocated(int, (K,))
        best_assigns = StridedArray.heap_allocated(int, (K,))

        @qy.for_(initializations)
        def _(i):
            @qy.for_(K)
            def _(k):
                # randomly assign the component
                j         = qy.random_int(N)
                component = StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1))

                j.store(assigns.at(k).data)

                self._sub_emitter.map(
                    prior.at(k),
                    samples.at(j).envelop(),
                    weights.at(j).envelop(),
                    component,
                    )

            # compute our total likelihood
            qy.value_from_any(0.0).store(total)

            @qy.for_(N)
            def _(n):
                sample = samples.at(n)

                mixture_ll = total.load()

                qy.value_from_any(-numpy.inf).store(total)

                @qy.for_(K)
                def _(k):
                    component_ll = total.load()

                    self._sub_emitter.ll(
                        StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                        sample,
                        total,
                        )

                    log_add_double(component_ll, total.load()).store(total)

                (mixture_ll + total.load()).store(total)

            # best observed so far?
            @qy.if_(total.load() >= best_ll.load())
            def _():
                total.load().store(best_ll)

                @qy.for_(K)
                def _(k):
                    assigns.at(k).data.load().store(best_assigns.at(k).data)

        # recompute the best observed assignment
        @qy.for_(K)
        def _(k):
            j = assigns.at(k).data.load()

            self._sub_emitter.ml(
                samples.at(j).envelop(),
                weights.at(j).envelop(),
                StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                )

        qy.heap_free(assigns.data)
        qy.heap_free(best_assigns.data)

        # generate random initial component weights
        @qy.for_(K)
        def _(k):
            r = qy.random()

            r.store(out.at(k).data.gep(0, 0))

            (total.load() + r).store(total)

        @qy.for_(K)
        def _(k):
            p = out.at(k).data.gep(0, 0)

            (p.load() / total.load()).store(p)

    def _map(self, prior, samples, weights, out, initializations):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        # mise en place
        K = self._model._K
        N = samples.shape[0]

        # generate some initial parameters
        self._map_initialize(prior, samples, weights, out, initializations)

        # run EM until convergence
        total        = qy.stack_allocate(float)
        component_ll = qy.stack_allocate(float)

        this_r_KN = StridedArray.heap_allocated(float, (K, N))
        last_r_KN = StridedArray.heap_allocated(float, (K, N))

        this_r_KN_data = Variable.set_to(this_r_KN.data)
        last_r_KN_data = Variable.set_to(last_r_KN.data)

        @qy.for_(self._model._iterations)
        def _(i):
            # compute responsibilities
            r_KN = this_r_KN.using(this_r_KN_data.value)

            @qy.for_(N)
            def _(n):
                sample = samples.at(n)

                qy.value_from_any(-numpy.inf).store(total)

                @qy.for_(K)
                def _(k):
                    responsibility = r_KN.at(k, n).data

                    self._sub_emitter.ll(
                        StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                        StridedArray.from_typed_pointer(sample.data),
                        responsibility,
                        )

                    log_add_double(total.load(), responsibility.load()).store(total)

                total_value = total.load()

                @qy.if_else(total_value == -numpy.inf)
                def _(then):
                    if then:
                        @qy.for_(K)
                        def _(k):
                            qy.value_from_any(1.0 / K).store(r_KN.at(k, n).data)
                    else:
                        @qy.for_(K)
                        def _(k):
                            responsibility = r_KN.at(k, n).data

                            qy.exp(responsibility.load() - total_value).store(responsibility)

            # estimate new mixture and component parameters
            @qy.for_(K)
            def _(k):
                component = out.at(k).data

                self._sub_emitter.map(
                    prior.at(k),
                    samples,
                    r_KN.at(k),
                    StridedArray.from_typed_pointer(component.gep(0, 1)),
                    )

                qy.value_from_any(0.0).store(total)

                @qy.for_(N)
                def _(n):
                    (total.load() + r_KN.at(k, n).data.load()).store(total)

                (total.load() / float(N)).store(component.gep(0, 0))

            # check for termination
            last_r_KN = this_r_KN.using(last_r_KN_data.value)

            @qy.if_(i > 0)
            def _():
                qy.value_from_any(0.0).store(total)

                @qy.for_(K)
                def _(k):
                    @qy.for_(N)
                    def _(n):
                        delta = r_KN.at(k, n).data.load() - last_r_KN.at(k, n).data.load()

                        (total.load() + abs(delta)).store(total)

                @qy.if_(total.load() < 1e-12)
                def _():
                    qy.break_()

            total_delta = total.load()

            # swap the responsibility matrices
            temp_r_KN_data_value = this_r_KN_data.value

            this_r_KN_data.set(last_r_KN_data.value)
            last_r_KN_data.set(temp_r_KN_data_value)

            # compute the ll at this step
            @qy.for_(N)
            def _(n):
                sample = samples.at(n)

                total_ll = total.load()

                qy.value_from_any(-numpy.inf).store(total)

                @qy.for_(K)
                def _(k):
                    self._sub_emitter.ll(
                        StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                        StridedArray.from_typed_pointer(sample.data),
                        component_ll,
                        )

                    log_add_double(
                        total.load(),
                        qy.log(out.at(k).data.gep(0, 0).load()) + component_ll.load(),
                        ) \
                        .store(total)

                (total_ll + total.load()).store(total)

            total_ll = total.load()

            # be informative
            qy.py_printf("after EM step %i: delta %s; ll %s\n", i, total_delta, total_ll)

        # clean up
        qy.heap_free(this_r_KN.data)
        qy.heap_free(last_r_KN.data)

        qy.return_()

    def given(self, parameter, samples, out):
        """
        Compute the conditional distribution.
        """

        @Function.define(
            Type.void(),
            [parameter.data.type_, samples.data.type_, out.data.type_],
            )
        def finite_mixture_given(parameter_data, samples_data, out_data):
            self._given(
                parameter.using(parameter_data),
                samples.using(samples_data),
                out.using(out_data),
                )

            qy.return_()

        finite_mixture_given(parameter.data, samples.data, out.data)

    def _given(self, parameter, samples, out):
        """
        Compute the conditional distribution.
        """

        # mise en place
        K = self._model._K
        N = samples.shape[0]

        # compute posterior mixture parameters
        total = qy.stack_allocate(float, -numpy.inf)

        @qy.for_(K)
        def _(k):
            prior_pi        = parameter.at(k).data.gep(0, 0)
            prior_parameter = parameter.at(k).data.gep(0, 1)
            posterior_pi    = out.at(k).data.gep(0, 0)

            qy.log(prior_pi.load()).store(posterior_pi)

            @qy.for_(N)
            def _(n):
                current_pi = posterior_pi.load()

                self._sub_emitter.ll(
                    StridedArray.from_typed_pointer(prior_parameter),
                    samples.at(n),
                    posterior_pi,
                    )

                (current_pi + posterior_pi.load()).store(posterior_pi)

            log_add_double(total.load(), posterior_pi.load()).store(total)

        total_value = total.load()

        @qy.for_(K)
        def _(k):
            posterior_pi  = out.at(k).data.gep(0, 0)
            normalized_pi = posterior_pi.load() - total_value

            qy.exp(normalized_pi).store(posterior_pi)

        # compute posterior component parameters
        @qy.for_(K)
        def _(k):
            prior_parameter     = parameter.at(k).data.gep(0, 1)
            posterior_parameter = out.at(k).data.gep(0, 1)

            self._sub_emitter.given(
                StridedArray.from_typed_pointer(prior_parameter),
                samples,
                StridedArray.from_typed_pointer(posterior_parameter),
                )

    def marginal(self, parameter, out):
        """
        Compute the marginal distribution.
        """

        @Function.define(
            Type.void(),
            [parameter.data.type_, out.data.type_],
            )
        def finite_mixture_marginal(parameter_data, out_data):
            self._marginal(
                parameter.using(parameter_data),
                out.using(out_data),
                )

        finite_mixture_marginal(parameter.data, out.data)

    def _marginal(self, parameter, out):
        """
        Compute the marginal distribution.
        """

        self._sub_emitter.average(
            parameter.extract(0, 0),
            parameter.extract(0, 1),
            out,
            )

        qy.return_()

