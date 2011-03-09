"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import absolute_import

import numpy
import contextlib

def tolist_deeply(value):
    """
    Fully convert a numpy object into a list.
    """

    if isinstance(value, numpy.ndarray):
        return map(tolist_deeply, value.tolist())
    elif isinstance(value, list):
        return map(tolist_deeply, value)
    elif isinstance(value, tuple):
        return tuple(map(tolist_deeply, value))
    else:
        return value

def normalize_dtype(dtype):
    """
    Construct an equivalent normal-form dtype.

    Normal-form dtypes are guaranteed to satisfy, in particular, the property
    of "shape greediness": the dtype's base property, if non-None, refers to a
    type with empty shape.
    """

    if dtype.shape:
        normal_base = normalize_dtype(dtype.base)

        return numpy.dtype((normal_base.base, dtype.shape + normal_base.shape))
    else:
        return dtype

def semicast(*arrays):
    """
    Broadcast compatible ndarray shape prefixes.
    """

    # establish the final prefix shape
    pre_ndim    = max(len(a.shape[:i]) for (a, i) in arrays)
    pre_padding = [(1,) * (pre_ndim - len(a.shape[:i])) for (a, i) in arrays]
    pre_shape   = tuple(map(max, *(p + a.shape[:i] for ((a, i), p) in zip(arrays, pre_padding))))

    # broadcast the arrays
    from numpy.lib.stride_tricks import as_strided

    casts = []

    for ((a, i), p) in zip(arrays, pre_padding):
        if i is None:
            i = len(a.shape)

        for (c, d) in zip(pre_shape[len(p):], a.shape[:i]):
            if c != d and d != 1:
                raise ValueError("array shapes incompatible for semicast")

        strides  = (0,) * len(p) + tuple(0 if d == 1 else s for (d, s) in zip(a.shape, a.strides))
        casts   += [as_strided(a, pre_shape + a.shape[i:], strides)]

    # repair dtypes (broken by as_strided)
    for ((a, _), cast) in zip(arrays, casts):
        cast.dtype = a.dtype

    # done
    return (pre_shape, casts)

def pretty_probability(p):
    return " C" if p >= 0.995 else "{0:02.0f}".format(p * 100.0)

def pretty_probability_row(row):
    return " ".join(map(pretty_probability, row))

def pretty_probability_matrix(matrix):
    return "\n".join(map(pretty_probability_row, matrix))

@contextlib.contextmanager
def numpy_printing(**kwargs):
    old = numpy.get_printoptions()

    numpy.set_printoptions(**kwargs)

    yield

    numpy.set_printoptions(**old)

