"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc         import abstractmethod
from cargo.sugar import ABC

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

