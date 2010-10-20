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
from cargo.log import get_logger

cimport numpy

from numpy cimport (
    ndarray,
    float_t,
    broadcast,
    int32_t,
    int64_t,
    )

cdef extern from "math.h":
    double log(double)

logger = get_logger(__name__)

cdef double log_add_double(double x, double y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic.
        Kingsbury and Rayner, 1970.
    """

    from math import (
        exp,
        log1p,
        )

    if x >= y:
        return x + log1p(exp(y - x))
    else:
        return y + log1p(exp(x - y))

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

        # prepare helper functions
        import ctypes

        from ctypes import sizeof

        uintptr_t = Type.int(sizeof(ctypes.c_void_p) * 8)
        log_t     = Type.pointer(Type.function(Type.double(), [Type.double()]))
        log_add_t = Type.pointer(Type.function(Type.double(), [Type.double(), Type.double()]))

        self._log     = Constant.int(uintptr_t, <long>&log).inttoptr(log_t)
        self._log_add = Constant.int(uintptr_t, <long>&log_add_double).inttoptr(log_add_t)

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
        mixing     = numpy.reshape(parameters["p"], extension)

        less   = 1 + len(self._distribution.parameter_dtype.shape)
        re_max = tuple([s - 1 for s in components.shape[:-less]])

        for i in numpy.ndindex(selected.shape):
            re_i = tuple(map(min, re_max, i))
            j    = numpy.nonzero(random.multinomial(1, mixing[re_i]))

            selected[i] = components[re_i + j]

        # generate random variates
        return self._distribution.rv(selected, out, random)

    def ll(self, builder, parameter_p, sample_p):
        """
        Compute finite-mixture log-likelihood.
        """

        # prepare the loop structure
        function = builder.basic_block.function

        start = builder.basic_block
        check = function.append_basic_block("finite_mixture_ll_loop_check")
        flesh = function.append_basic_block("finite_mixture_ll_loop_flesh")
        leave = function.append_basic_block("finite_mixture_ll_loop_leave")

        builder.branch(check)

        # build the check block
        builder.position_at_end(check)

        index_type = Type.int(32)
        total_type = Type.double()
        zero_index = Constant.int(index_type, 0)
        one_index  = Constant.int(index_type, 1)
        index      = builder.phi(index_type, "index")
        total      = builder.phi(total_type, "total")

        index.add_incoming(zero_index, start)
        index.add_incoming(builder.add(index, one_index), flesh)
        total.add_incoming(
            Constant.real(total_type, numpy.finfo(numpy.float64).min),
            start,
            )

        builder.cbranch(
            builder.icmp(
                llvm.core.ICMP_UGT,
                Constant.int(index_type, self._model._K),
                index,
                ),
            flesh,
            leave,
            )

        # build the flesh block
        builder.position_at_end(flesh)

        component_p   = builder.gep(parameter_p, [zero_index, index, zero_index])
        component_sum = \
            builder.add(
                builder.call(
                    self._log,
                    [builder.load(builder.gep(component_p, [zero_index, zero_index]))],
                    ),
                self._sub_emitter.ll(
                    builder,
                    builder.gep(component_p, [zero_index, one_index]),
                    sample_p,
                    ),
                )
        next_total = builder.call(self._log_add, [total, component_sum])

        total.add_incoming(next_total, flesh)

        builder.branch(check)

        # wrap up the loop
        builder.position_at_end(leave)

        return total

    def ml(
                                   self,
        ndarray                    samples, # ndim = 2
        ndarray[float_t, ndim = 2] weights,
        ndarray                    out,     # ndim = 1
                                   random = numpy.random,
        ):
        """
        Use EM to estimate mixture parameters.
        """

        # arguments
        assert samples.shape[0] == weights.shape[0]
        assert samples.shape[1] == weights.shape[1]

        if not numpy.all(weights == 1.0):
            raise NotImplementedError("non-unit sample weighting not yet supported")

        if out is None:
            out = numpy.empty(samples.shape[0], self._parameter_dtype)
        else:
            assert samples.shape[0] == out.shape[0]

        # computation
        logger.detail("estimating finite mixture from %i samples" % samples.shape[1])

        for i in xrange(samples.shape[0]):
            out[i] = self._ml(samples[i], weights[i], random)

        return out

    def _ml(
                                   self,
        ndarray                    samples_N,
        ndarray[float_t, ndim = 1] weights_N,
                                   random = numpy.random,
        ):
        """
        Use EM to estimate mixture parameters.
        """

        # mise en place
        cdef size_t N = samples_N.shape[0]
        cdef size_t K = self._K

        d = self._distribution
        p = numpy.empty((), self._parameter_dtype)

        # generate a random initial state
        seeds = random.randint(N, size = K)

        d.ml(samples_N[seeds][:, None], weights_N[seeds][:, None], p["c"], random)

        p["p"]  = random.rand(K)
        p["p"] /= numpy.sum(p["p"])

        # run EM until convergence
        last_r_KN = None
        r_KN      = numpy.empty((K, N))

        for i in xrange(self._iterations):
            # evaluate responsibilities
            d.ll(p["c"][:, None], samples_N, r_KN)

            numpy.exp(r_KN, r_KN)

            r_KN *= p["p"][:, None]
            r_KN /= numpy.sum(r_KN, 0)

            # make maximum-likelihood estimates
            d.ml(samples_N, r_KN, p["c"], random)

            p["p"] = numpy.sum(r_KN, 1) / N

            # check for convergence
            if last_r_KN is None:
                last_r_KN = numpy.empty((K, N))
            else:
                difference = numpy.sum(numpy.abs(r_KN - last_r_KN))

                logger.detail(
                    "iteration %i < %i ; delta %e >? %e",
                    i,
                    self._iterations,
                    difference,
                    self._convergence,
                    )

                if difference < self._convergence:
                    break

            (last_r_KN, r_KN) = (r_KN, last_r_KN)

        # done
        return p

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

