"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

# FIXME really shouldn't expose LLVM types in the AST

class Constant(object):
    """
    The trivial fixed constant distribution.
    """

    def __init__(self, type_):
        """
        Initialize.
        """

        # we can only support (for now) types with simple equality tests
        if type_.kind == llvm.core.TYPE_DOUBLE:
            pass
        else:
            raise ValueError("unsupported type for constant distribution")

        self._type = type_

    def for_module(self, module):
        """
        Return a specialized builder.
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

    def ll(self, builder, parameter, sample):
        """
        Compute constant-distribution log-likelihood.
        """

        return \
            builder.select(
                builder.fcmp(
                    llvm.core.FCMP_OEQ,
                    parameter,
                    sample,
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
    def sample_type(self):
        """
        Sample dtype.
        """

        return self._type

    @property
    def parameter_type(self):
        """
        Parameter dtype.
        """

        return self._type

