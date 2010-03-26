"""
cargo/statistics/distribution.py

The distribution ABC.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from abc import abstractmethod
from numpy import newaxis
from cargo.log import get_logger
from cargo.sugar import ABC

log = get_logger(__name__)

class Estimator(ABC):
    """
    Estimate a distribution from samples.
    """

    @abstractmethod
    def estimate(self, samples):
        """
        Return the estimated distribution.
        """

        pass

