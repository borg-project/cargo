# cython: profile=True
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from cargo.statistics.base import (
    Estimator,
    Distribution,
    SampleSequence,
    )

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

        return [d.random_variate(random) for d in self._inner]

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

        cdef int    i
        cdef double p = 0.0

        for i in xrange(len(self._inner)):
            p += self._inner[i].log_likelihood(sample[i])

        return p

    def add_log_likelihoods(self, samples, to):
        """
        Add the log likelihoods of C{samples} under this distribution.
        """

        if not isinstance(samples, TupleSamples):
            samples = TupleSamples.from_sequence(samples)

        cdef int i

        for i in xrange(len(self._inner)):
            self._inner[i].add_log_likelihoods(samples._sequences[i], to)

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of C{samples} under this distribution.
        """

        if not isinstance(samples, TupleSamples):
            samples = TupleSamples.from_sequence(samples)

        if len(samples._sequences) != len(self._inner):
            raise ValueError("samples and distribution width do not match")

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

        if len(samples._sequences) != len(self._estimators):
            raise \
                ValueError(
                    "samples width %i does not match estimators count %i" % (
                        len(samples._sequences),
                        len(self._estimators),
                        ),
                    )

        from itertools import izip

        zipped = izip(self._estimators, samples._sequences)

        return TupleDistribution([e.estimate(s, random, weights) for (e, s) in zipped])

def tuple_samples_from_sequence(samples):
    """
    Build a tuple-specific sample sequence from a generic sample sequence.
    """

    return TupleSamples(zip(*samples))

class TupleSamples(SampleSequence):
    """
    Store samples from a tuple distribution.
    """

    def __init__(self, sequences):
        """
        Initialize.

        @param sequences: Sequence of samples from each component.
        """

        # sanity
        for i in xrange(len(sequences) - 1):
            if not isinstance(sequences[i], SampleSequence):
                raise TypeError("inner sequence is not an array or sequence")

            if len(sequences[i]) != len(sequences[i + 1]):
                raise ValueError("sample sequence lengths are mismatched")

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

        return TupleSamplesIterator(self)

    def __getitem__(self, index):
        """
        Return a particular sample.
        """

        return [s[index] for s in self._sequences]

    def __str__(self):
        """
        Return a string description of this sequence.
        """

        return str(list(self))

    def __repr__(self):
        """
        Return a string representation of this sequence.
        """

        return repr(list(self))

    from_sequence = tuple_samples_from_sequence

class TupleSamplesIterator(object):
    """
    Iterate over tuple samples.
    """

    def __init__(self, samples):
        """
        Initialize.
        """

        self._samples = samples
        self._i       = 0

    def next(self):
        """
        Return the next sample.
        """

        cdef int i = self._i

        if i >= len(self._samples):
            raise StopIteration()
        else:
            self._i = i + 1

            return [s[i] for s in self._samples._sequences]

