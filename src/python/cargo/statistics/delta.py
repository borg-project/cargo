"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import llvm.core

from llvm.core import (
    Type,
    Constant,
    )
from cargo.llvm.high_level import high

class Delta(object):
    """
    The trivial fixed constant distribution.
    """

    def __init__(self, dtype):
        """
        Initialize.
        """

        from cargo.llvm import type_from_dtype

        self._dtype = numpy.dtype(dtype)
        self._type  = type_from_dtype(self._dtype)

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

    def ll(self, parameter, sample, out):
        """
        Compute constant-distribution log-likelihood.
        """

        high.select(
            parameter.data.load() == sample.data.load(),
            0.0,
            numpy.finfo(float).min,
            ) \
        .store(out)

    def given(self, parameter, samples, out):
        """
        Compute the posterior distribution.
        """

        parameter.data.load().store(out.data)

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

