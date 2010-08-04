"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from contextlib import (
    nested,
    closing,
    )

def chunk(sequence, size):
    """
    Yield a sequence of chunks from sequence.
    """

    from itertools import groupby

    for (_, pairs) in groupby(enumerate(sequence), lambda (i, _): i / size):
        yield (v for (_, v) in pairs)

def shuffled(sequence, random = numpy.random):
    """
    Return an array copy of the sequence in random order.
    """

    copy = numpy.copy(sequence)

    random.shuffle(copy)

    return copy

def closing_all(*args):
    """
    Return a context manager closing the passed arguments.
    """

    return nested(*[closing(f) for f in args])

def replace_all(string, *args):
    """
    Return the result of successive string replacements.
    """

    for replacement in args:
        string = string.replace(*replacement)

    return string

def iflatten(v):
    """
    Yield elements from an iterable, recursing into all inner iterables.

    @see: iflatten_in
    @see: iflatten_ex
    """

    return iflatten_ex(v)

def iflatten_in(v, r = ()):
    """
    Yield elements from an iterable, recursing only into a certain types.

    @param v: An iterable of elements to yield.
    @param r: A tuple of types into which to recurse.
    """

    if isinstance(v, r):
        for e in v:
            for f in iflatten_in(e, r):
                yield f
    else:
        yield v

def iflatten_ex(v, nr = ()):
    """
    Yield elements from an iterable, recursing not into a certain types.

    @param v: An iterable of elements to yield.
    @param nr: A tuple of types not into which to recurse.
    """

    if hasattr(v, "__iter__") and not isinstance(v, nr):
        for e in v:
            for f in iflatten_ex(e, nr):
                yield f
    else:
        yield v

def non_none(*args):
    """
    Return the first non-None value in the arguments list, or None.
    """

    try:
        return (a for a in args if a is not None).next()
    except StopIteration:
        return None

