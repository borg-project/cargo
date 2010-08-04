"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.statistics.base import Distribution

class Constant(Distribution):
    """
    The trivial fixed constant distribution.
    """

    def __init__(self, constant):
        """
        Initialize.
        """

        self._constant = constant

    def random_variate(self, random = numpy.random):
        """
        Return the constant.
        """

        return self._constant

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        if sample == self._constant:
            return 0.0
        else:
            return numpy.finfo(float).min

