"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from numpy                     import (
    ones,
    array,
    )
from numpy.random              import RandomState
from nose.tools                import (
    assert_not_equal,
    assert_almost_equal,
    )
from cargo.statistics          import (
    Delta,
    ModelEngine,
    FiniteMixture,
    MixedBinomial,
    )

def assert_finite_mixture_ml_ok(me):
    """
    Verify EM estimation of finite mixture distributions.
    """

    from cargo.testing import assert_almost_equal_deep

    (e,) = \
        me.ml(
            [[(7, 8)] * 100 + [(1, 8)] * 200],
            ones((1, 300)),
            )

    print e

    assert_almost_equal_deep(
        e[numpy.argsort(e["p"])].tolist(),
        [(1.0 / 3.0, 7.0 / 8.0),
         (2.0 / 3.0, 1.0 / 8.0)],
        places = 4,
        )

def test_finite_mixture_ml():
    """
    Test EM estimation of finite mixture distributions.
    """

    me = ModelEngine(FiniteMixture(MixedBinomial(epsilon = 0.0), 2))

    assert_finite_mixture_ml_ok(me)

def test_finite_mixture_ll():
    """
    Test finite-mixture log-likelihood computation.
    """

    engine = ModelEngine(FiniteMixture(Delta(float), 2))
    p = [[(0.25, 1.0), (0.75, 2.0)]]

    assert_almost_equal(engine.ll(p,   1.0 ), numpy.log(0.25))
    assert_almost_equal(engine.ll(p, [ 2.0]), numpy.log(0.75))
    assert_almost_equal(engine.ll(p, [42.0]), numpy.finfo(float).min)

def test_finite_mixture_rv():
    """
    Test finite-mixture random-variate generation.
    """

    d = FiniteMixture(Delta(float), 2)
    p = numpy.array([[(0.25, 1.0), (0.75, 2.0)]], d.parameter_dtype.base)
    s = numpy.empty(32768)

    d.rv(p, s, RandomState(42))

    assert_almost_equal(s[s == 1.0].size / float(s.size), 0.25, places = 2)
    assert_almost_equal(s[s == 2.0].size / float(s.size), 0.75, places = 2)

def test_finite_mixture_given():
    """
    Test finite-mixture posterior-parameter computation.
    """

    model  = FiniteMixture(Delta(float), 2)
    engine = ModelEngine(model)
    out    = engine.given([(0.25, 1.0), (0.75, 2.0)], [2.0])

    assert_almost_equal(out["p"][0], 0.0)
    assert_almost_equal(out["p"][1], 1.0)

def test_restarting_ml():
    """
    Test the restarting-ML distribution wrapper.
    """

    from cargo.statistics.mixture  import RestartingML
    from cargo.statistics.binomial import MixedBinomial

    m = FiniteMixture(MixedBinomial(epsilon = 0.0), 2)
    d = RestartingML(m)

    assert_finite_mixture_ml_ok(d)

