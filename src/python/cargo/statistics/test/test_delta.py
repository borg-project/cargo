"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_constant():
    """
    Test the trivial constant distribution.
    """

    import numpy

    from nose.tools       import assert_equal
    from cargo.statistics import (
        Delta,
        ModelEngine,
        )

    me = ModelEngine(Delta(numpy.float64))

    #assert_equal(constant.random_variate(), 42.0)
    assert_equal(me.ll(42.0, 42.1), numpy.finfo(float).min)
    assert_equal(me.ll(42.0, 42.0), 0.0)

