"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

numpy.seterr(divide = "raise", invalid = "raise", over = "warn", under = "warn") # FIXME hack

def log_add_scalar(x, y):
    """
    Return log(x + y) given log(x) and log(y); see [1].

    [1] Digital Filtering Using Logarithmic Arithmetic.
        Kingsbury and Rayner, 1970.
    """

    from math import (
        exp,
        log1p,
        )

    if x >= y:
        return x + log1p(exp(y - x))
    else:
        return y + log1p(exp(x - y))

log_add = numpy.frompyfunc(log_add_scalar, 2, 1)

