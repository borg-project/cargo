"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from collections import namedtuple
from llvm.core   import (
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

AA            = \
ArrayArgument = \
    namedtuple(
        "ArrayArgument",
        [
            "array",
            "dtype",
            "extra",
            ],
        )

def semicast_arguments(out_argument, *in_arguments):
    """
    Appropriately broadcast arguments and an output array.
    """

    # semicast arrays
    from cargo.numpy import semicast

    prefix_i = lambda d, e : -(e + len(d.shape)) or None
    out_pair = (out_argument.array, prefix_i(*out_argument[1:]))
    in_pairs = [(numpy.asarray(a, d.base), prefix_i(d, e)) for (a, d, e) in in_arguments]

    if out_argument.array is None:
        (shape, cast_arrays) = semicast(*in_pairs)
        out_array            = numpy.empty(shape[:out_pair[0]], out_argument.dtype)

        return (shape, [out_array] + cast_arrays)
    else:
        (shape, cast_arrays) = semicast(*([out_pair] + in_pairs))

        assert out.shape == shape[:out_pair[0]] + out_argument.dtype.shape

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
                AA(out       , numpy.dtype(numpy.float64) , 0),
                AA(parameters, self._model.parameter_dtype, 0),
                AA(samples   , self._model.sample_dtype   , 0),
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

                emitter.ll(l.arrays["p"], l.arrays["s"], l.arrays["o"].data)

        # done
        return out

    def ml(self, samples, weights, out = None, random = numpy.random):
        """
        Compute the estimated maximum-likelihood parameter.
        """

        # arguments
        (shape, (out, samples, weights)) = \
            semicast_arguments(
                AA(out    , self._model.parameter_dtype, 0),
                AA(samples, self._model.sample_dtype   , 1),
                AA(weights, numpy.dtype(numpy.float64) , 1),
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
            emitter = self._model.get_emitter(module)

            @arrays.loop_all(len(shape))
            def _(l):
                emitter.ml(l.arrays["s"], l.arrays["w"], l.arrays["o"])

        # done
        return out

    def given(self, parameters, samples, out = None):
        """
        Compute the posterior parameter.
        """

        # arguments
        (shape, (out, parameters, samples)) = \
            semicast_arguments(
                AA(out       , self._model.parameter_dtype, 0),
                AA(parameters, self._model.parameter_dtype, 0),
                AA(samples   , self._model.sample_dtype   , 1),
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
            emitter = self._model.get_emitter(module)

            @arrays.loop_all(len(shape))
            def _(l):
                print "parameter array shape", l.arrays["p"].shape
                print "samples array shape", l.arrays["s"].shape
                print "out array shape", l.arrays["o"].shape

                emitter.given(l.arrays["p"], l.arrays["s"], l.arrays["o"])

        # done
        return out

    @property
    def model(self):
        """
        Return the model.
        """

        return self._model

