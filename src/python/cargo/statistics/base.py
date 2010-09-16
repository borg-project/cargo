"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from collections import Sequence
from cargo.sugar import ABC

class SampleSequence(ABC):
    """
    Interface to a sequence of samples.

    Does not provide methods beyond that of Sequence; its purpose is to tie in
    additional supported types that also provide the Sequence interface, such
    as numpy.ndarray.
    """

SampleSequence.register(Sequence)
SampleSequence.register(numpy.ndarray)

class Distribution(object):
    """
    Interface to a probability distribution.
    """

    def random_variate(self, random = numpy.random):
        """
        Return a single sample from this distribution.
        """

        raise NotImplementedError()

    def random_variates(self, size, random = numpy.random):
        """
        Return a sequence of independent samples from this distribution.

        @return: A collection of values that supports the Sequence interface.
        """

        return [self.random_variate(random) for i in xrange(size)]

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.

        @param samples: An arbitrary sample value.
        """

        raise NotImplementedError()

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of C{samples} under this distribution.

        @param samples: A value supporting the Sequence interface.
        """

        return sum(self.log_likelihood(s) for s in samples)

    def given(self, samples):
        """
        Return a conditional distribution.

        The meaning of "conditioned" depends on the distribution, but is
        nontrivial only for hierarchical distributions: it assumes that future
        samples will be drawn from the same leaf distribution as were past
        samples.
        """

        return self

class Estimator(object):
    """
    Interface to a maximum-likelihood estimator of distributions.
    """

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        raise NotImplementedError()

