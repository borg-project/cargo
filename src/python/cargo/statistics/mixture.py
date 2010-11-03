"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys
import numpy
import llvm.core

from llvm.core import (
    Type,
    Constant,
    )
from cargo.log             import get_logger
from cargo.llvm            import StridedArray
from cargo.llvm.high_level import (
    high,
    HighFunction,
    )

logger = get_logger(__name__)

def log_add_double(x, y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic.
        Kingsbury and Rayner, 1970.
    """

    b  = x >= y
    x_ = high.select(b, x, y)
    y_ = high.select(b, y, x)

    return x_ + high.log1p(high.exp(y_ - x_))

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

    def get_emitter(self, module):
        """
        Return an IR emitter for this distribution.
        """

        return FiniteMixtureEmitter(self, module)

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

        return self._parameter_dtype

    @property
    def sample_dtype(self):
        """
        Return the sample type.
        """

        return self._distribution.sample_dtype

class FiniteMixtureEmitter(object):
    """
    Emit IR for the FiniteMixture distribution.
    """

    def __init__(self, model, module):
        """
        Initialize.
        """

        self._model       = model
        self._module      = module
        self._sub_emitter = self._model.distribution.get_emitter(module)

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

        # XXX eeek leaking stack space
        total        = high.stack_allocate(Type.double(), numpy.finfo(float).min)
        component_ll = high.stack_allocate(Type.double())

        @high.for_(self._model._K)
        def _(index):
            component = parameter.at(index)

            self._sub_emitter.ll(
                StridedArray.from_typed_pointer(component.data.gep(0, 1)),
                sample,
                component_ll,
                )

            #@high.python(total.load(), component_ll.load())
            #def _(ll_py, cl_py):
                #print "total ll", ll_py
                #print "component ll", cl_py

            log_add_double(
                total.load(),
                high.log(component.data.gep(0, 0).load()) + component_ll.load(),
                ) \
                .store(total)

            #@high.python(component.data.gep(0, 0).load(), total.load())
            #def _(pi_py, ll_py):
                #print "log pi", pi_py
                #print "total ll", ll_py

        total.load().store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        from cargo.llvm import StridedArray

        # mise en place
        K = self._model._K
        N = samples.shape[0]

        # generate a random initial state
        # XXX eeek leaking stack space
        total = high.stack_allocate(float, 0.0)

        @high.for_(K)
        def _(k):
            n = high.random_int(N)

            self._sub_emitter.ml(
                samples.at(n).envelop(),
                weights.at(n).envelop(),
                StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                )

            r = high.random()

            r.store(out.at(k).data.gep(0, 0))

            (total.load() + r).store(total)

        @high.for_(K)
        def _(k):
            p = out.at(k).data.gep(0, 0)

            (p.load() / total.load()).store(p)

        # run EM until convergence
        r_KN = StridedArray.heap_allocated(float, (K, N)) # XXX never deallocated

        @high.for_(self._model._iterations)
        def _(i):
            # compute responsibilities
            @high.for_(N)
            def _(n):
                sample = samples.at(n)

                high.value_from_any(0.0).store(total)

                @high.for_(K)
                def _(k):
                    responsibility = r_KN.at(k, n).data

                    self._sub_emitter.ll(
                        StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
                        StridedArray.from_typed_pointer(sample.data),
                        responsibility,
                        )

                    high.exp(responsibility.load()).store(responsibility)

                    #@high.python(total.load(), responsibility.load())
                    #def _(total_py, r_py):
                        #print "total", total_py, "r", r_py

                    (total.load() + responsibility.load()).store(total)

                total_value = total.load()

                @high.for_(K)
                def _(k):
                    responsibility = r_KN.at(k, n).data

                    (responsibility.load() / total_value).store(responsibility)

            # make maximum-likelihood estimates
            @high.for_(K)
            def _(k):
                component = out.at(k).data

                self._sub_emitter.ml(
                    samples,
                    r_KN.at(k),
                    StridedArray.from_typed_pointer(component.gep(0, 1)),
                    )

                high.value_from_any(0.0).store(total)

                @high.for_(samples.shape[0])
                def _(n):
                    (total.load() + r_KN.at(k, n).data.load()).store(total)

                (total.load() / float(N)).store(component.gep(0, 0))

    def given(self, parameter, samples, out):
        """
        Return the conditional distribution.
        """

        # mise en place
        K = self._model._K
        N = samples.shape[0]

        # compute posterior mixture parameters
        # XXX eeek leaking stack space
        total = high.stack_allocate(float, 0.0)

        @high.for_(K)
        def _(k):
            component           = out.at(k).data
            component_pi        = component.gep(0, 0)
            component_parameter = parameter.at(k).data.gep(0, 1)

            high.log(parameter.at(k).data.gep(0, 0).load()).store(component_pi)

            # XXX clean up above, out / parameter distinction

            @high.for_(N)
            def _(n):
                previous = component_pi.load()

                self._sub_emitter.ll(
                    StridedArray.from_typed_pointer(component_parameter),
                    samples.at(n),
                    component_pi,
                    )

                (component_pi.load() + previous).store(component_pi)

            exped = high.exp(component_pi.load())

            exped.store(component_pi)

            (total.load() + exped).store(total)

        @high.for_(K)
        def _(k):
            component_pi = out.at(k).data.gep(0, 0)

            (component_pi.load() / total.load()).store(component_pi)

        @high.for_(K)
        def _(k):
            self._sub_emitter.given(
                StridedArray.from_typed_pointer(parameter.at(k).data.gep(0, 1)),
                samples,
                StridedArray.from_typed_pointer(out.at(k).data.gep(0, 1)),
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

