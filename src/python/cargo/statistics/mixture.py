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

    exp   = HighFunction("exp"  , float, [float])
    log1p = HighFunction("log1p", float, [float])

    return \
        high.select(
            x >= y,
            x + log1p(exp(y - x)),
            y + log1p(exp(x - y)),
            )

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
                K,
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

        log = HighFunction("log", float, [float])

        # XXX eeek leaking stack space
        total        = high.stack_allocate(Type.double())
        component_ll = high.stack_allocate(Type.double())

        high.value(numpy.finfo(float).min).store(total)

        @high.for_(self._model._K)
        def _(index):
            component = parameter.at(index)

            self._sub_emitter.ll(component.gep(0, 1), sample, component_ll)

            log_add_double(
                total.load(),
                log(component.gep(0, 0).load()) + component_ll.load(),
                ) \
                .store(total)

        total.load().store(out)

    def ml(self, samples, weights, out):
        """
        Emit computation of the estimated maximum-likelihood parameter.
        """

        from cargo.llvm import StridedArray

        # mise en place
        K   = self._model._K
        N   = samples.shape[0]
        exp = HighFunction("exp", float, [float])

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

            @high.python(k, r)
            def _(k_py, r_py):
                print k_py, r_py

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

                high.value(0.0).store(total)

                @high.for_(K)
                def _(k):
                    responsibility = r_KN.at(k, n).data

                    self._sub_emitter.ll(out.at(k).data.gep(0, 1), sample.data, responsibility)

                    exp(responsibility.load()).store(responsibility)

                    (total.load() + responsibility.load()).store(total)

                total_value = total.load()

                @high.for_(K)
                def _(k):
                    responsibility = r_KN.at(k, n).data

                    (responsibility.load() / total_value).store(responsibility)

                    #@high.python(n, k, responsibility.load())
                    #def _(n_py, k_py, r_py):
                        #print "responsibility of %s for %s is %s" % (k_py, n_py, r_py)

            # make maximum-likelihood estimates
            @high.for_(K)
            def _(k):
                component = out.at(k).data

                self._sub_emitter.ml(
                    samples,
                    r_KN.at(k),
                    StridedArray.from_typed_pointer(component.gep(0, 1)),
                    )

                high.value(0.0).store(total)

                @high.for_(samples.shape[0])
                def _(n):
                    (total.load() + r_KN.at(k, n).data.load()).store(total)

                (total.load() / float(N)).store(component.gep(0, 0))

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

