"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import absolute_import

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
    return casts

