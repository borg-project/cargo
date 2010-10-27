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
from cargo.llvm import (
    emit_and_execute,
    StridedArrays,
    )

numpy.seterr(divide = "raise", invalid = "raise", over = "warn", under = "warn") # FIXME hack

def semicast_arguments(out_options, *pairs):
    """
    Appropriately broadcast arguments and an output array.
    """

    if len(out_options) == 2:
        (out, out_dtype) = out_options
        out_shrink       = None
    elif len(out_options) == 3:
        (out, out_dtype, out_shrink) = out_options
    else:
        raise ValueError("bad out options")

    from cargo.numpy import semicast

    array_pairs   = [(numpy.asarray(v, d.base), d) for (v, d) in pairs]
    pairs_to_cast = [(a, -len(d.shape) or None) for (a, d) in array_pairs]

    if out is None:
        (shape, cast_arrays) = semicast(*pairs_to_cast)

        return (shape, [numpy.empty(shape[:out_shrink], out_dtype)] + cast_arrays)
    else:
        (shape, cast_arrays) = semicast(*((out, out_dtype) + pairs_to_cast))

        assert out.shape == shape[:out_shrink]

        return (shape, cast_arrays)

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
        (shape, (out, parameters, samples)) = \
            semicast_arguments(
                (out       , numpy.dtype(numpy.float64) ),
                (parameters, self._model.parameter_dtype),
                (samples   , self._model.sample_dtype   ),
                )

        # computation
        @emit_and_execute("local")
        def _(module):
            """
            Emit the log-likelihood computation.
            """

            arrays = \
                StridedArrays.from_numpy({
                    "p" : parameters,
                    "s" : samples,
                    "o" : out,
                    })

            @arrays.loop_all(len(shape))
            def _(l):
                emitter = self._model.get_emitter(module)

                emitter.ll(l.arrays["p"], l.arrays["s"], l.arrays["o"])

        # done
        return out

    def ml(self, samples, weights, out = None, random = numpy.random):
        """
        Return the estimated maximum-likelihood parameter.
        """

        # arguments
        (shape, (out, samples, weights)) = \
            semicast_arguments(
                (out    , self._model.parameter_dtype, -1),
                (samples, self._model.sample_dtype       ),
                (weights, numpy.dtype(numpy.float64)     ),
                )

        # computation
        @emit_and_execute("local")
        def _(module):
            """
            Emit the log-likelihood computation.
            """

            arrays = \
                StridedArrays.from_numpy({
                    "s" : samples,
                    "w" : weights,
                    "o" : out,
                    })

            @arrays.loop_all(len(shape) - 1)
            def _(l):
                emitter = self._model.get_emitter(module)

                emitter.ml(l.arrays["s"], l.arrays["w"], l.arrays["o"])

        # done
        return out

    @property
    def model(self):
        """
        Return the model.
        """

        return self._model

