"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

def get_random_random(random = numpy.random):
    """
    Get a randomly-initialized PRNG.
    """

    from numpy.random import RandomState

    return RandomState(random.randint(numpy.iinfo(int).max))

def random_subsets(sequence, sizes, random = numpy.random):
    """
    Return a series of non-intersecting random subsets of a sequence.
    """

    sa      = shuffled(sequence, random = random)
    index   = 0
    subsets = []

    for size in sizes:
        assert len(sa) >= index + size

        subsets.append(sa[index:index + size])

        index += size

    return subsets

def draw(p, random = numpy.random, normalize = True):
    """
    Return an index selected according to array of probabilities C{p}.

    Normalizes by default.
    """

    if normalize:
        p = p / numpy.sum(p)

    ((i,),) = numpy.nonzero(random.multinomial(1, p))

    return i

def grab(sequence, random = numpy.random):
    """
    Return a randomly-selected element from the sequence.
    """

    return sequence[random.randint(len(sequence))]

