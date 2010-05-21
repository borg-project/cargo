"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy.random

def get_random_random(random = numpy.random):
    """
    Get a randomly-initialized PRNG.
    """

    from numpy        import iinfo
    from numpy.random import RandomState

    return RandomState(random.randint(iinfo(int).max))

