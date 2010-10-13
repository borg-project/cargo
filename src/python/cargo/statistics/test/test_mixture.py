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
from cargo.log                 import get_logger
from cargo.testing             import assert_almost_equal_deep
from cargo.statistics.mixture  import FiniteMixture

def test_finite_mixture_ml():
    """
    Test EM estimation of finite mixture distributions.
    """

    from cargo.statistics.binomial import MixedBinomial

    get_logger("cargo.statistics.mixture", level = "NOTSET")

    d    = FiniteMixture(MixedBinomial(epsilon = 0.0), 2)
    (e,) = \
        d.ml(
            array([[(7, 8)] * 100 + [(1, 8)] * 200], d.sample_dtype),
            ones((1, 300)),
            None,
            RandomState(41),
            )

    assert_almost_equal_deep(
        e[numpy.argsort(e["p"])].tolist(),
        [(1.0 / 3.0, 7.0 / 8.0),
         (2.0 / 3.0, 1.0 / 8.0)],
        places = 4,
        )

def test_finite_mixture_ll():
    """
    Test finite-mixture log-likelihood computation.
    """

    from cargo.statistics.constant import Constant

    d = FiniteMixture(Constant(numpy.float_), 2)
    p = numpy.array([[(0.25, 1.0), (0.75, 2.0)]], d.parameter_dtype.base)

    assert_almost_equal(d.ll(p,   1.0 ), numpy.log(0.25))
    assert_almost_equal(d.ll(p, [ 2.0]), numpy.log(0.75))
    assert_almost_equal(d.ll(p, [42.0]), numpy.finfo(float).min)

def test_finite_mixture_rv():
    """
    Test finite-mixture random-variate generation.
    """

    # build a simple finite mixture
    from cargo.statistics.constant import Constant

    d = FiniteMixture(Constant(numpy.float_), 2)
    p = numpy.array([[(0.25, 1.0), (0.75, 2.0)]], d.parameter_dtype.base)

    # test sampling
    s = numpy.empty(32768)

    d.rv(p, s, RandomState(42))

    assert_almost_equal(s[s == 1.0].size / float(s.size), 0.25, places = 2)
    assert_almost_equal(s[s == 2.0].size / float(s.size), 0.75, places = 2)

def test_finite_mixture():
    """
    Test a finite mixture.
    """

    # build a simple finite mixture
    from cargo.statistics.mixture  import FiniteMixture
    from cargo.statistics.constant import Constant

    one     = Constant(1.0)
    two     = Constant(2.0)
    mixture = FiniteMixture([0.25, 0.75], [one, two])

    # test sampling
    from nose.tools import (
        assert_not_equal,
        assert_almost_equal,
        )

    # test conditional distribution computation
    def test_given():
        """
        Test computation of a posterior finite mixture.
        """

        conditional = mixture.given([2.0])

        assert_almost_equal(conditional.pi[0], 0.0)
        assert_almost_equal(conditional.pi[1], 1.0)

    yield test_given

def assert_mixture_estimator_ok(estimator):
    """
    Test estimation of finite mixture distributions.
    """

    # generate some data
    from numpy.random                 import RandomState
    from cargo.statistics.multinomial import Multinomial

    random     = RandomState(42)
    components = [Multinomial([0.1, 0.9], 8), Multinomial([0.9, 0.1], 8)]
    samples    = [components[0].random_variate(random) for i in xrange(250)]
    samples   += [components[1].random_variate(random) for i in xrange(750)]

    # estimate the distribution from data
    import numpy

    from nose.tools import assert_almost_equal

    mixture   = estimator.estimate(samples, random = random)
    order     = numpy.argsort(mixture.pi)
    estimated = numpy.asarray(mixture.components)[order]

    assert_almost_equal(mixture.pi[order][0], 0.25, places = 2)
    assert_almost_equal(mixture.pi[order][1], 0.75, places = 2)
    assert_almost_equal(estimated[0].beta[0], 0.10, places = 2)
    assert_almost_equal(estimated[0].beta[1], 0.90, places = 2)
    assert_almost_equal(estimated[1].beta[0], 0.90, places = 2)
    assert_almost_equal(estimated[1].beta[1], 0.10, places = 2)

def test_em_mixture_estimator():
    """
    Test EM estimation of finite mixture distributions.
    """

    from cargo.statistics.mixture     import FiniteMixture
    from cargo.statistics.multinomial import Multinomial

    estimator = FiniteMixture([Multinomial()] * 2)

    assert_mixture_estimator_ok(estimator)

def test_restarted_estimator():
    """
    Test the restarting wrapper estimator.
    """

    from cargo.statistics.mixture     import (
        RestartedEstimator,
        EM_MixtureEstimator,
        )
    from cargo.statistics.multinomial import MultinomialEstimator

    estimator = EM_MixtureEstimator([MultinomialEstimator()] * 2)
    restarted = RestartedEstimator(estimator, 3)

    assert_mixture_estimator_ok(restarted)

