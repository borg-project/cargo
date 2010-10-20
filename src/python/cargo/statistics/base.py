"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

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

def dtype_from_integer_type(type_):
    """
    Build a numpy dtype from an LLVM integer type.
    """

    sizes = {
        8  : numpy.dtype(numpy.int8),
        16 : numpy.dtype(numpy.int16),
        32 : numpy.dtype(numpy.int32),
        64 : numpy.dtype(numpy.int64),
        }

    return sizes[type_.width]

def dtype_from_struct_type(type_):
    """
    Build a numpy dtype from an LLVM struct type.
    """

    fields = [
        ("f%i" % i, dtype_from_type(f))
        for (i, f) in enumerate(type_.elements)
        ]

    return numpy.dtype(fields)

def dtype_from_type(type_):
    """
    Build a numpy dtype from an LLVM type.
    """

    from llvm import core

    mapping = {
        core.TYPE_FLOAT   : (lambda _ : numpy.dtype(numpy.float32)),
        core.TYPE_DOUBLE  : (lambda _ : numpy.dtype(numpy.float64)),
        core.TYPE_INTEGER : dtype_from_integer_type,
        core.TYPE_STRUCT  : dtype_from_struct_type,
        }

    return mapping[type_.kind](type_)

class ModelEngine(object):
    """
    Operations on a model.
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
        from cargo.numpy import semicast

        parameters = numpy.asarray(parameters, self._model.parameter_dtype.base)
        samples    = numpy.asarray(samples   , self._model.sample_dtype.base   )

        if out is None:
            (shape, (parameters, samples)) = \
                semicast(
                    (parameters, -len(self._model.parameter_dtype.shape) or None),
                    (samples   , -len(self._model.sample_dtype.shape)    or None),
                    )

            out = numpy.empty(shape, numpy.float64)
        else:
            (shape, (parameters, samples, _)) = \
                semicast(
                    (parameters, -len(self._model.parameter_dtype.shape) or None),
                    (samples   , -len(self._model.sample_dtype.shape)    or None),
                    (out       ,                                            None),
                    )

            assert out.shape == parameters.shape

        # prepare for code generation
        from llvm.ee   import (
            TargetData,
            ExecutionEngine,
            )
        from llvm.core import (
            Type,
            Module,
            Builder,
            Constant,
            )

        local   = Module.new("distribution_ll")
        emitter = self._model.get_emitter(local)
        main    = local.add_function(Type.function(Type.void(), []), "main")
        entry   = main.append_basic_block("entry")
        builder = Builder.new(entry)

        # build the computation
        from cargo.statistics.lowloop import strided_array_loop

        def emit_ll_call(builder, locations):
            """
            Emit the body of the array loop.
            """

            builder.store(
                emitter.ll(
                    builder,
                    locations["p"],
                    locations["s"],
                    ),
                locations["o"],
                )

        strided_array_loop(
            builder,
            emit_ll_call,
            shape,
            {
                "p" : parameters,
                "s" : samples,
                "o" : out,
                },
            "ll_loop",
            )

        builder.ret_void()

        # compile and execute
        print local

        local.verify()

        engine = ExecutionEngine.new(local)

        engine.run_function(main, [])

        # done
        return out

    def ml(self, sam_loop, weight_loop, out_p, prng):
        """
        Return the estimated maximum-likelihood parameter.
        """

        raise NotImplementedError()

    @property
    def model(self):
        """
        Return the model.
        """

        return self._model

