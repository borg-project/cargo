"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import llvm.core

from llvm.core import (
    Type,
    Constant,
    )

# FIXME really shouldn't expose LLVM types in the AST

class Delta(object):
    """
    The trivial fixed constant distribution.
    """

    def __init__(self, dtype):
        """
        Initialize.
        """

        from cargo.statistics.lowloop import dtype_to_type

        self._dtype = numpy.dtype(dtype)
        self._type  = dtype_to_type(self._dtype)

        # we can only support (for now) types that provide simple equality tests
        supported = set([
            llvm.core.TYPE_DOUBLE,
            ])

        if self._type.kind not in supported:
            raise ValueError("unsupported type for constant distribution")

    def get_emitter(self, module):
        """
        Return an IR emitter for this distribution.
        """

        return self

    def rv(self, parameters, out, random = numpy.random):
        """
        Return the constant.
        """

        # arguments
        parameters = numpy.asarray(parameters, self._dtype)

        if out is None:
            out = numpy.empty(parameters.shape)
        else:
            print parameters.shape, out.shape, "incompat?"
            (parameters, out) = numpy.broadcast_arrays(parameters, out)

            if out.dtype != numpy.float_:
                raise ValueError("out argument has invalid dtype")

        # computation
        out[:] = parameters

        return out

    def ll(self, builder, parameter_p, sample_p):
        """
        Compute constant-distribution log-likelihood.
        """

        return \
            builder.select(
                builder.fcmp(
                    llvm.core.FCMP_OEQ,
                    builder.load(parameter_p),
                    builder.load(sample_p),
                    ),
                Constant.real(Type.double(), 0.0),
                Constant.real(Type.double(), numpy.finfo(numpy.float_).min),
                )

    def given(self, parameters, samples, out = None):
        """
        Return the conditional distribution.
        """

        from cargo.numpy import semicast

        parameters = numpy.asarray(parameters, self._dtype)
        samples    = numpy.asarray(samples   , self._dtype)

        if out is None:
            (parameters, samples) = \
                semicast(
                    (parameters, None),
                    (samples   , None),
                    )

            out = numpy.empty(samples.shape, dtype = self._parameter_dtype)
        else:
            (parameters, samples, _) = \
                semicast(
                    (parameters, None),
                    (samples   , None),
                    (out       , None),
                    )

            assert out.shape == parameters.shape

        out[:] = parameters

        return out

    @property
    def sample_dtype(self):
        """
        Sample dtype.
        """

        return self._dtype

    @property
    def parameter_dtype(self):
        """
        Parameter dtype.
        """

        return self._dtype
