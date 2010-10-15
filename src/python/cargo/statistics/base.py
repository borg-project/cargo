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

def dtype_from_type(type_):
    """
    Build a numpy dtype from an LLVM type.
    """

    raise NotImplementedError()

class Distribution(object):
    """
    Operations on a distribution.
    """

    def __init__(self, sample_):
        """
        Initialize.
        """

        self._LowDistribution = LowDistribution

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

        parameters = numpy.asarray(parameters, self.parameter_dtype.base)
        samples    = numpy.asarray(samples   , self.sample_dtype.base   )

        if out is None:
            (shape, (parameters, samples)) = \
                semicast(
                    (parameters, -len(self.parameters_dtype.shape) or None),
                    (samples   , -len(self.sample_dtype.shape)     or None),
                    )

            out = numpy.empty(shape, numpy.float64)
        else:
            (shape, (parameters, samples, _)) = \
                semicast(
                    (parameters, -len(self.parameters_dtype.shape) or None),
                    (samples   , -len(self.sample_dtype.shape)     or None),
                    (out       ,                                      None),
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

        local = Module.new("distribution_ll")
        core  = self.core(local)
        main  = local.add_function(Type.function(Type.void(), []), "main")
        entry = main.append_basic_block("entry")
        exit  = main.append_basic_block("exit")

        # build the computation
        loop = ArrayLoop(main, shape, exit, {"p" : parameters, "s" : samples, "o" : out})

        core.ll(loop.builder, loop.locations["p"], loop.locations["s"], loop.locations["o"])

        # build the entry blocks
        entry_builder = Builder.new(entry)

        entry_builder.branch(loop.entry_block)

        # build the exit block
        exit_builder = Builder.new(exit)

        exit_builder.ret_void()

        # compile and compute
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
    def low(self):
        """
        Return the low-level distribution compiler class.
        """

        return self._LowDistribution

    @property
    def parameter_dtype(self):
        """
        Type of a distribution parameter.
        """

        return self._parameter_dtype

    @property
    def sample_dtype(self):
        """
        Type of a sample.
        """

        return self._sample_dtype

