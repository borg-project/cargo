"""
utexas/statistics/functions.py

General functions in statistics.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import scipy

from numpy import (
    log,
    exp)
from cargo.statistics._statistics import (
    add_log,
    pochhammer_ln,
    )

#In [2]: pochhammer_ln(1, 2)
#Out[2]: 0.69314718055994529

#In [3]: pochhammer_ln(2, 3)
#Out[3]: 3.1780538303479458

"""
Return the log of the Pochhammer symbol (x)_n.
"""

