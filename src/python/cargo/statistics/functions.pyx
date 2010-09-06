"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef extern from "math.h":
    long double expl(long double)
    long double logl(long double)

cpdef long double add_log(long double x, long double y):
    """
    Return log(x + y) given log(x) and log(y).

    Sacrifices accuracy for (mildly eccentric) sanity.
    """

    if x == 0.0:
        return y
    elif y == 0.0:
        return x
    elif x - y > 16.0:
        return x
    elif x > y:
        return x + logl(1.0 + expl(y - x))
    elif y - x > 16.0:
        return y
    else:
        return y + logl(1.0 + expl(x - y))

