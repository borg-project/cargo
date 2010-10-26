"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from llvm.core import (
    Type,
    Module,
    Builder,
    Constant,
    )

numpy.seterr(divide = "raise", invalid = "raise", over = "warn", under = "warn") # FIXME hack

def log_add_scalar(x, y):
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

log_add = numpy.frompyfunc(log_add_scalar, 2, 1)

def semicast_arguments((out, out_dtype), *pairs):
    """
    Appropriately broadcast arguments and an output array.
    """

    from cargo.numpy import semicast

    array_pairs   = [(numpy.asarray(v, d.base), d) for (v, d) in pairs]
    pairs_to_cast = [(a, -len(d.shape) or None) for (a, d) in array_pairs]

    if out is None:
        (shape, cast_arrays) = semicast(*pairs_to_cast)

        return (shape, cast_arrays + [numpy.empty(shape, out_dtype)])
    else:
        (shape, cast_arrays) = semicast(*(pairs_to_cast + (out, out_dtype)))

        assert out.shape == shape

        return (shape, cast_arrays)

def emit_and_execute(module):
    """
    Prepare for, emit, and run some LLVM IR.
    """

    from cargo.llvm import this_builder

    def decorator(emitter):
        # emit some IR
        main    = module.add_function(Type.function(Type.void(), []), "main")
        entry   = main.append_basic_block("entry")

        with this_builder(Builder.new(entry)) as builder:
            emitter()

            builder.ret_void()

        # then compile and execute it
        from llvm.ee import ExecutionEngine

        print module

        module.verify()

        engine = ExecutionEngine.new(module)

        engine.run_function(main, [])

    return decorator

class ModelEngine(object):
    """
    Vectorized operations on a model.
    """

    def __init__(self, model):
        """
        Initialize.
        """

        self._model = model

    def rv(self, b, par_p, out_p, prng):
        """
        Return samples from this distribution.
        """

        raise NotImplementedError()

    def ll(self, parameters, samples, out = None):
        """
        Compute log probability under this distribution.
        """

        # arguments
        (shape, (parameters, samples, out)) = \
            semicast_arguments(
                (out       , numpy.dtype(numpy.float64) ),
                (parameters, self._model.parameter_dtype),
                (samples   , self._model.sample_dtype   ),
                )

        # computation
        module  = Module.new("local")
        emitter = self._model.get_emitter(module)

        @emit_and_execute(module)
        def _():
            """
            Emit the log-likelihood computation.
            """

            from cargo.llvm.lowloop import StridedArrays

            arrays = \
                StridedArrays.from_numpy({
                    "p" : parameters,
                    "s" : samples,
                    "o" : out,
                    })

            @arrays.loop_all(len(shape))
            def _(l):
                emitter.ll(l.arrays["p"], l.arrays["s"], l.arrays["o"])

        # done
        return out

    def ml(self, samples, weights, out = None, axis = -1, random = numpy.random):
        """
        Return the estimated maximum-likelihood parameter.
        """

        ## arguments
        #sample_shape = self._model.sample_dtype.shape
        #real_axis    = if axis >= 0 then axis else axis - len(sample_shape)
        #samples_axis_dtype = numpy.dtype(self._model.parameter_dtype, 

        #(shape, (parameters, samples)) = \
            #semicast_arguments(
                #(out       , self._model.parameter_dtype),
                #(samples   , self._model.sample_dtype   ),
                #)
        #(shape, (parameters, samples, out)) = \
            #semicast_arguments(
                #(out       , numpy.dtype(numpy.float64) ),
                #(parameters, self._model.parameter_dtype),
                #(samples   , self._model.sample_dtype   ),
                #)

    @property
    def model(self):
        """
        Return the model.
        """

        return self._model

