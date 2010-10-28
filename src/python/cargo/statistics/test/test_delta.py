"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from nose.tools       import assert_equal
from cargo.statistics import (
    Delta,
    ModelEngine,
    )

def test_delta_ll():
    """
    Test likelihood computation under the trivial delta distribution.
    """

    engine = ModelEngine(Delta(float))

    assert_equal(engine.ll(42.0, 42.1), numpy.finfo(float).min)
    assert_equal(engine.ll(42.0, 42.0), 0.0)

def test_delta_rv():
    """
    Test random variate generation under the trivial delta distribution.
    """

    engine = ModelEngine(Delta(float))

    assert_equal(engine.rv(), 42.0)

def test_delta_given():
    """
    Test posterior computation under the trivial delta distribution.
    """

    engine = ModelEngine(Delta(float))

    assert_equal(engine.given(42.0, [43.0]), 42.0)

