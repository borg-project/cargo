"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc         import abstractmethod
from collections import Sequence
from cargo.sugar import ABC

def set_up_gsl():
    """
    Set up the GSL.
    """

    from cargo.statistics._dcm import disable_gsl_error_handler

    disable_gsl_error_handler()

set_up_gsl()

class SampleSequence(ABC):
    """
    Interface to a sequence of samples.

    Does not provide methods beyond that of Sequence; its purpose is to tie in
    additional supported types that also provide the Sequence interface, such
    as numpy.ndarray.
    """

SampleSequence.register(Sequence)
SampleSequence.register(numpy.ndarray)

class Distribution(ABC):
    """
    Interface to a probability distribution.
    """

    @abstractmethod
    def random_variate(self, random = numpy.random):
        """
        Return a single sample from this distribution.
        """

    def random_variates(self, size, random = numpy.random):
        """
        Return a sequence of independent samples from this distribution.

        @return: A collection of values that supports the Sequence interface.
        """

        return [self.random_variate(random) for i in xrange(size)]

    @abstractmethod
    def log_likelihood(self, sample):
        """
        Return the log likelihood of C{sample} under this distribution.

        @param samples: An arbitrary sample value.
        """

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

class Estimator(ABC):
    """
    Interface to a maximum-likelihood estimator of distributions.
    """

    @abstractmethod
    def estimate(self, samples, random = numpy.random, weights = None):
        """
        Return the estimated distribution.
        """

    @staticmethod
    def build(request):
        """
        Build an estimator as requested.
        """

        from cargo.statistics import (
            DCM_Estimator,
            TupleEstimator,
            RestartedEstimator,
            EM_MixtureEstimator,
            MultinomialEstimator,
            )

        builders = {
            "dcm"         : DCM_Estimator.build,
            "tuple"       : TupleEstimator.build,
            "restarted"   : RestartedEstimator.build,
            "mixture"     : EM_MixtureEstimator.build,
            "multinomial" : MultinomialEstimator.build,
            }

        return builders[request["type"]](request)
