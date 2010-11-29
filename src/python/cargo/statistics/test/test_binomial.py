"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy            import array
from nose.tools       import assert_almost_equal
from cargo.testing    import assert_almost_equal_deep
from cargo.statistics import (
    Binomial,
    ModelEngine,
    MixedBinomial,
    )

def test_binomial_ll():
    """
    Test log-probability computation in the binomial distribution.
    """

    me = ModelEngine(Binomial())

    assert_almost_equal(me.ll((0.25, 2), 1), -0.98082925)
    assert_almost_equal_deep(
        me.ll(
            [[(0.25, 2), (0.25, 5)],
             [(0.75, 4), (0.75, 8)]],
            [[1, 4],
             [3, 8]],
            ),
        [[-0.98082925, -4.22342160],
         [-0.86304622, -2.30145658]],
        )

def test_binomial_ml():
    """
    Test max-likelihood estimation under the binomial distribution.
    """

    me = ModelEngine(Binomial(estimation_n = 2))

    assert_almost_equal_deep(
        me.ml(
            [[1, 2],
             [2, 0]],
            numpy.ones((2, 2)),
            ) \
            .tolist(),
        [(0.75, 2), (0.50, 2)],
        )

def test_binomial_map():
    """
    Test MAP estimation under the binomial distribution.
    """

    me = ModelEngine(Binomial(estimation_n = 2))

    assert_almost_equal_deep(
        me.map(
            [(1   , 1   ),
             (0.25, 0.75),
             (2   , 3   )],
            [[1, 2],
             [1, 2],
             [1, 2]],
            numpy.ones((3, 2)),
            ) \
            .tolist(),
        [(0.75     , 2),
         (0.75     , 2),
         (4.0 / 7.0, 2)],
        )

def test_mixed_binomial_ll():
    """
    Test log-probability computation in the mixed binomial distribution.
    """

    me = ModelEngine(MixedBinomial())

    assert_almost_equal(me.ll(0.25, (1, 2)), -0.98082925)
    assert_almost_equal_deep(
        me.ll(
            [[0.25],
             [0.75]],
            [[(1, 2), (4, 5)],
             [(3, 4), (8, 8)]],
            ),
        [[-0.98082925, -4.22342160],
         [-0.86304622, -2.30145658]],
        )

def test_mixed_binomial_ml():
    """
    Test max-likelihood estimation under the mixed binomial distribution.
    """

    me = ModelEngine(MixedBinomial())

    assert_almost_equal_deep(
        me.ml(
            [[(1, 2), (4, 5)],
             [(3, 4), (8, 8)]],
            numpy.ones((2, 2)),
            ),
        [5.0 / 7.0, 11.0 / 12.0],
        )

def test_mixed_binomial_map():
    """
    Test MAP estimation under the mixed binomial distribution.
    """

    me = ModelEngine(MixedBinomial())

    assert_almost_equal_deep(
        me.map(
            [(1   , 1   ),
             (0.25, 0.75),
             (2   , 3   )],
            [[(1, 2), (2, 2)],
             [(1, 2), (2, 2)],
             [(1, 2), (2, 2)]],
            numpy.ones((3, 2)),
            ) \
            .tolist(),
        [0.75, 0.75, 4.0 / 7.0],
        )

