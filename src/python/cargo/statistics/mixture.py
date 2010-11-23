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
    StridedArray,
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

    def get_emitter(self):
        """
        Return an IR emitter for this distribution.
        """

        return FiniteMixtureEmitter(self)

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
        #mixing = numpy.reshape(parameters["p"], extension)
        #less   = 1 + len(self._distribution.parameter_dtype.shape)
        #re_max = tuple([s - 1 for s in components.shape[:-less]])

        #for i in numpy.ndindex(selected.shape):
            #re_i = tuple(map(min, re_max, i))
            #j    = numpy.nonzero(random.multinomial(1, mixing[re_i]))

            #selected[i] = components[re_i + j]

        ## generate random variates
        #return self._distribution.rv(selected, out, random)

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

        total        = qy.stack_allocate(Type.double(), -numpy.inf, "total")
        component_ll = qy.stack_allocate(Type.double())

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

            qy.return_()

        finite_mixture_ml(samples.data, weights.data, out.data)

    def _ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        # mise en place
        K = self._model._K
        N = samples.shape[0]

        # generate a random initial state
        total = qy.stack_allocate(float, 0.0)

        @qy.for_(K)
        def _(k):
            n = qy.random_int(N)

            self._sub_emitter.ml(
                samples.at(n).envelop(),
                weights.at(n).envelop(),
                StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                )

            r = qy.random()

            r.store(out.at(k).data.gep(0, 0))

            (total.load() + r).store(total)

        @qy.for_(K)
        def _(k):
            p = out.at(k).data.gep(0, 0)

            (p.load() / total.load()).store(p)

        # run EM until convergence
        r_KN = StridedArray.heap_allocated(float, (K, N)) # XXX leaking heap space

        @qy.for_(self._model._iterations)
        def _(i):
            # compute responsibilities
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

                self._sub_emitter.ml(
                    samples,
                    r_KN.at(k),
                    StridedArray.from_typed_pointer(component.gep(0, 1)),
                    )

                qy.value_from_any(0.0).store(total)

                @qy.for_(samples.shape[0])
                def _(n):
                    (total.load() + r_KN.at(k, n).data.load()).store(total)

                (total.load() / float(N)).store(component.gep(0, 0))

            #@qy.if_(i % 16 == 0)
            #def _():
            qy.py_printf("completed EM iteration %i\n", i)

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

        @qy.if_else(total_value == -numpy.inf)
        def _(then):
            if then:
                @qy.for_(K)
                def _(k):
                    qy.value_from_any(1.0 / K).store(out.at(k).data.gep(0, 0))
            else:
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

#class RestartingML(object):
    #"""
    #Wrap a distribution with a restarting ML estimator.
    #"""

    #def __init__(self, distribution, restarts = 8):
        #"""
        #Initialize.
        #"""

        #self._distribution = distribution
        #self._restarts     = restarts

    #def rv(self, parameters, out, random = numpy.random):
        #"""
        #Make a draw from this mixture distribution.
        #"""

        #return self._distribution.rv(parameters, out, random)

    #def ll(self, parameters, samples, out = None):
        #"""
        #Compute finite-mixture log-likelihood.
        #"""

        #return self._distribution.ll(parameters, samples, out)

    #def ml(self, samples, weights, out, random = numpy.random):
        #"""
        #Use EM to estimate mixture parameters.
        #"""

        #raise NotImplementedError()

    #def given(self, parameters, samples, out = None):
        #"""
        #Return the conditional distribution.
        #"""

        #return self._distribution.given(parameters, samples, out)

    #@property
    #def distribution(self):
        #"""
        #Return the mixture components.
        #"""

        #return self._distribution

    #@property
    #def parameter_dtype(self):
        #"""
        #Return the parameter type.
        #"""

        #return self._distribution.parameter_dtype

    #@property
    #def sample_dtype(self):
        #"""
        #Return the sample type.
        #"""

        #return self._distribution.sample_dtype

