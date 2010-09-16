"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef extern from "math.h":
    double      exp (double)
    double      log (double)
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

cpdef double log_plus(double x, double y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic.
        Kingsbury and Rayner, 1970.
    """

    if x >= y:
        return x + log(1.0 + exp(y - x))
    else:
        return y + log(1.0 + exp(x - y))

