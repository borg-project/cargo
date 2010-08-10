"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from collections           import Sequence
from cargo.statistics.base import (
    Estimator,
    Distribution,
    )

class TupleSamples(Sequence):
    """
    Store samples from a tuple distribution.
    """

    def __init__(self, sequences):
        """
        Initialize.

        @param sequences: Sequence of samples from each component.
        """

        # sanity
#         for i in xrange(len(sequences) - 1):
#             if not isinstance(sequences[i], Sequence):
#                 raise TypeError("inner sequence is not a sequence")

#             if len(sequences[i]) != len(sequences[i + 1]):
#                 raise ValueError("sample sequence lengths are mismatched")

        # members
        self._sequences = sequences

    def __len__(self):
        """
        How many samples are stored?
        """

        return len(self._sequences[0])

    def __iter__(self):
        """
        Return an iterator over the stored samples.
        """

        for i in xrange(len(self)):
            yield self[i]

    def __getitem__(self, index):
        """
        Return a particular sample.
        """

        return tuple(s[index] for s in self._sequences)

    @staticmethod
    def from_sequence(samples):
        """
        Build a tuple-specific sample sequence from a generic sample sequence.
        """

        return TupleSamples(zip(*samples))

#class SparseTupleSample(object):
    #"""
    #Sparsely store a single sample from a tuple distribution.
    #"""

    #def __init__(self, indices, samples):
        #"""
        #Initialize.
        #"""

        #self._indices = indices
        #self._samples = samples

    #def _log_likelihood(self, distribution):
        #"""
        #Return the log likelihood of this sample under C{distribution}.
        #"""

        #from itertools import izip

        #return sum(distribution._inner[i] for (i, s) in izip(self._indices, self._samples))

class TupleDistribution(Distribution):
    """
    A tuple of independent distributions.
    """

    def __init__(self, distributions):
        """
        Initialize.
        """

        self._inner = distributions

    def random_variate(self, random = numpy.random):
        """
        Return a single sample from this distribution.
        """

        return tuple(d.random_variate(random) for d in self._inner)

    def random_variates(self, size, random = numpy.random):
        """
        Return a sequence of samples from this distribution.

        @param size: The size of the sample set to return.
        """

        return TupleSamples([d.random_variates(size, random) for d in self._inner])

    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.
        """

        from itertools import izip

        return sum(d.log_likelihood(s) for (d, s) in izip(self._inner, sample))

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of C{samples} under this distribution.
        """

        if not isinstance(samples, TupleSamples):
            samples = TupleSamples.from_sequence(samples)

        from itertools import izip

        zipped = izip(self._inner, samples._sequences)

        return sum(d.total_log_likelihood(s) for (d, s) in zipped)

    @property
    def inner(self):
        """
        Return the inner distributions.
        """

        return self._inner

class TupleEstimator(Estimator):
    """
    A maximum-likelihood estimator of tuple distributions.
    """

    def __init__(self, estimators):
        """
        Initialize.
        """

        self._estimators = estimators

    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

        if not isinstance(samples, TupleSamples):
            samples = TupleSamples.from_sequence(samples)

        from itertools import izip

        zipped = izip(self._estimators, samples._sequences)

        return TupleDistribution([e.estimate(s, random, weights) for (e, s) in zipped])

    @staticmethod
    def build(request):
        """
        Build a tuple-distribution estimator as requested.
        """

        return TupleEstimator(map(Estimator.build, request["estimators"]))

