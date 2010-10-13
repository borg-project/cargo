"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

class Constant(object):
    """
    The trivial fixed constant distribution.
    """

    def __init__(self, dtype):
        """
        Initialize.
        """

        self._dtype = numpy.dtype(dtype)

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

    def ll(self, parameters, samples, out = None):
        """
        Compute constant-distribution log-likelihood.
        """

        # arguments
        parameters = numpy.asarray(parameters, self._dtype)
        samples    = numpy.asarray(samples,    self._dtype)

        (parameters, samples) = numpy.broadcast_arrays(parameters, samples)

        if out is None:
            out = numpy.empty(parameters.shape)
        else:
            if out.shape != parameters.shape:
                raise ValueError("out argument has invalid shape")
            if out.dtype != numpy.float_:
                raise ValueError("out argument has invalid dtype")

        # computation
        out[:] = numpy.finfo(numpy.float_).min
        out[samples == parameters] = 0.0

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

