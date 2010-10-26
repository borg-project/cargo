# cython: profile=True
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

cimport numpy

from numpy cimport (
    ndarray,
    float_t,
    broadcast,
    int32_t,
    int64_t,
    uint8_t,
    )

cdef extern from "math.h":
    double log(double)

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

cdef void put_double(double v):
    sys.stdout.write("%s" % v)

cdef void put_int(int32_t v):
    sys.stdout.write("%s" % v)

cdef void put_string(char* string):
    sys.stdout.write(string)

def emit_print_string(builder, string):
    from cargo.llvm import iptr_type

    print_string_t = Type.pointer(Type.function(Type.void(), [Type.pointer(Type.int(8))]))
    print_string   = Constant.int(iptr_type, <long>&put_string).inttoptr(print_string_t)

    from llvm.core import GlobalVariable

    module    = builder.basic_block.function.module
    cstring   = GlobalVariable.new(module, Type.array(Type.int(8), len(string) + 1), "cstring")
    cstring_p = builder.gep(cstring, [Constant.int(Type.int(32), 0)] * 2)

    cstring.initializer = Constant.stringz(string)

    builder.call(print_string, [cstring_p])

def emit_print(builder, *values):
    import ctypes

    from ctypes     import sizeof
    from cargo.llvm import iptr_type

    print_double_t = Type.pointer(Type.function(Type.void(), [Type.double()]))
    print_int_t    = Type.pointer(Type.function(Type.void(), [Type.int(32)]))

    print_double = Constant.int(iptr_type, <long>&put_double).inttoptr(print_double_t)
    print_int    = Constant.int(iptr_type, <long>&put_int).inttoptr(print_int_t)

    for value in values:
        if value.type.kind == llvm.core.TYPE_DOUBLE:
            builder.call(print_double, [value])
        elif value.type.kind == llvm.core.TYPE_INTEGER:
            assert value.type.width == 32

            builder.call(print_double, [value])

        emit_print_string(builder, " ")

    emit_print_string(builder, "\n")

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

#cdef struct CFiniteMixture:
    #uint32_t K
    #uint32_t iterations
    #double   convergence

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

    def rv(self, parameters, out, random = numpy.random):
        """
        Make a draw from this mixture distribution.
        """

        # identify the common prefix
        if self._distribution.sample_dtype.shape:
            out_prefix = out.shape[:-len(self._distribution.sample_dtype.shape)]
        else:
            out_prefix = out.shape

        selected = numpy.empty(out_prefix, dtype = self._distribution.parameter_dtype)

        # select the relevant components
        extension  = (1,) * (len(selected.shape) - len(parameters.shape)) + parameters.shape
        components = \
            numpy.reshape(
                parameters["c"],
                extension + self._distribution.parameter_dtype.shape,
                )
        mixing = numpy.reshape(parameters["p"], extension)
        less   = 1 + len(self._distribution.parameter_dtype.shape)
        re_max = tuple([s - 1 for s in components.shape[:-less]])

        for i in numpy.ndindex(selected.shape):
            re_i = tuple(map(min, re_max, i))
            j    = numpy.nonzero(random.multinomial(1, mixing[re_i]))

            selected[i] = components[re_i + j]

        # generate random variates
        return self._distribution.rv(selected, out, random)

    def ll(self, parameter, sample, out):
        """
        Compute finite-mixture log-likelihood.
        """

        log = HighFunction("log", float, [float])

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

    #def ml(self, samples, weights, out):
        #"""
        #Emit computation of the estimated maximum-likelihood parameter.
        #"""

        ## generate a random initial state
        #builder = high.builder
        #p_total = builder.alloca(Type.double())

        #p_total.store_value(0)

        #@high.for_(self._K)
        #def _(k):
            ##n = random.randint(N)
            #n = 0
            #self._sub_emitter.ml(sample.at(n), weight.at(n), out.at(k).gep(0, 1))

            ##p["p"]  = random.rand(K)
            #out.at(k).gep(0, 0).store_value(0.1)
            #(p_total.load() + 0.1).store(p_total)

        #@high.for_(self._K)
        #def _(k):
            #p = out.at(k).gep(0, 0)

            #(p.load() / p_total.load()).store(p)

        # run EM until convergence
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

                #logger.detail(
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

    def given(self, parameters, samples, out = None):
        """
        Return the conditional distribution.
        """

        # arguments
        from cargo.numpy import semicast

        parameters = numpy.asarray(parameters, self._parameter_dtype.base)
        samples    = numpy.asarray(samples   , self.sample_dtype         )

        if out is None:
            (parameters, samples) = \
                semicast(
                    (parameters, -1                                   ),
                    (samples   , -len(self.sample_dtype.shape) or None),
                    )

            print parameters.shape, samples.shape

            out = numpy.empty_like(parameters)
        else:
            (parameters, samples, _) = \
                semicast(
                    (parameters, -1                                   ),
                    (samples   , -len(self.sample_dtype.shape) or None),
                    (out       , -1                                   ),
                    )

            assert out.shape == parameters.shape

        # compute posterior mixture parameters
        out["p"]  = parameters["p"]

        ll = self._distribution.ll(parameters["c"], samples[..., None])

        if ll.ndim > 1:
            sum_ll = numpy.sum(ll, -2)
        else:
            sum_ll = ll

        out["p"] *= numpy.exp(sum_ll)
        out["p"] /= numpy.sum(out["p"], -1)[..., None]

        # compute posterior mixture components
        self._distribution.given(parameters["c"], samples[..., None], out["c"])

        # done
        return out

#cdef struct StridedAxis:
    #uint8_t* data
    #uint32_t dimensionality
    #uint32_t stride

#cdef void ml_springboard(
    #PyObject* settings,
    #uint32_t        N       ,
    #StridedAxis     samples ,
    #StridedAxis     weights ,
    #StridedAxis     outs    ,
    #):

#cdef void estimate_ml_parameter(
    #CFiniteMixture* settings,
    #uint32_t        N       ,
    #StridedAxis     samples ,
    #StridedAxis     weights ,
    #StridedAxis     outs    ,
    #):
    #"""
    #Use EM to estimate mixture parameters.
    #"""

    ### mise en place
    #cdef size_t K = self._K

    #d = self._distribution

    ## generate a random initial state
    #seeds = random.randint(N, size = K)

    #for i in xrange(...):
        #n = random.randint(N)
        #inner.ml(sample + n * stride, weight + n * stride, out["c"])

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

            #logger.detail(
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

