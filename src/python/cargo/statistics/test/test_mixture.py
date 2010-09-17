"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_finite_mixture():
    """
    Test a finite mixture.
    """

    # build a simple finite mixture
    import numpy

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

    def test_random_variate():
        """
        Test sampling of a random variate from a finite mixture.
        """

        from numpy.random import RandomState

        ones   = 0
        twos   = 0
        random = RandomState(42)
        draws  = 32768

        for i in xrange(draws):
            sample = mixture.random_variate(random = random)

            if sample == 1.0:
                ones += 1
            elif sample == 2.0:
                twos += 1
            else:
                assert_not_equal(sample, sample)

        assert_almost_equal(ones / float(draws), 0.25, places = 2)
        assert_almost_equal(twos / float(draws), 0.75, places = 2)

    yield test_random_variate

    # test likelihood computation
    def test_log_likelihood():
        """
        Test computation of log likelihood in a finite mixture.
        """

        assert_almost_equal(mixture.log_likelihood( 1.0), numpy.log(0.25))
        assert_almost_equal(mixture.log_likelihood( 2.0), numpy.log(0.75))
        assert_almost_equal(mixture.log_likelihood(42.0), numpy.finfo(float).min)

    yield test_log_likelihood

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

    from cargo.statistics.mixture     import EM_MixtureEstimator
    from cargo.statistics.multinomial import MultinomialEstimator

    estimator = EM_MixtureEstimator([MultinomialEstimator()] * 2)

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

