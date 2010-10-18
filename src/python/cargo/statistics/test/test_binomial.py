"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy                     import (
    ones,
    array,
    )
from nose.tools                import assert_almost_equal
from cargo.testing             import assert_almost_equal_deep
from cargo.statistics          import (
    Binomial,
    Distribution,
    )
#from cargo.statistics.binomial import MixedBinomial

def test_binomial_ll():
    """
    Test log-probability computation in the binomial distribution.
    """

    d = Distribution(Binomial())

    assert_almost_equal(d.ll((0.25, 2), 1), -0.98082925)
    assert_almost_equal_deep(
        d.ll(
            [[(0.25, 2), (0.25, 5)],
             [(0.75, 4), (0.75, 8)]],
            [[1, 4,],
             [3, 8,]],
            ),
        [[-0.98082925, -4.22342160],
         [-0.86304622, -2.30145658]],
        )

def test_mixed_binomial_ll():
    """
    Test log-probability computation in the mixed binomial distribution.
    """

    d = MixedBinomial()

    #assert_almost_equal_deep(
        #d.ll(
            #[[0.25],
             #[0.75]],
            #[[(1, 2), (4, 5)],
             #[(3, 4), (8, 8)]],
            #),
        #[[-0.98082925, -4.22342160],
         #[-0.86304622, -2.30145658]],
        #)
    assert_almost_equal(d.ll(0.25, (1, 2)), -0.98082925)

def test_mixed_binomial_ml():
    """
    Test log-probability computation in the mixed binomial distribution.
    """

    d = MixedBinomial(epsilon = 0.0)

    assert_almost_equal_deep(
        d.ml(
            [[(1, 2), (4, 5)],
             [(3, 4), (8, 8)]],
            ones((2, 2)),
            ),
        [5.0 / 7.0, 11.0 / 12.0],
        )

