"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_finite_mixture():
    """
    Test a finite mixture.
    """

    # build a simple finite mixture
    import numpy

    from cargo.statistics.mixture      import FiniteMixture
    from cargo.statistics.distribution import ConstantDistribution

    one     = ConstantDistribution(1.0)
    two     = ConstantDistribution(2.0)
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

        for i in xrange(1024):
            sample = mixture.random_variate()

            if sample == 1.0:
                ones += 1
            elif sample == 2.0:
                twos += 1
            else:
                assert_not_equal(sample, sample)

        assert_almost_equal(ones / 1024.0, 0.25)
        assert_almost_equal(twos / 1024.0, 0.75)

    yield test_random_variate

    # test likelihood computation
    def test_log_likelihood():
        """
        Test computation of log likelihood in a finite mixture.
        """

        from math import log

        assert_almost_equal(mixture.log_likelihood(1.0), log(0.25))
        assert_almost_equal(mixture.log_likelihood(2.0), log(0.75))
        assert_almost_equal(mixture.log_likelihood(42.0), numpy.finfo(float).min)

    yield test_log_likelihood

def test_em_mixture_estimator():
    """
    Test EM estimation of finite mixture distributions.
    """

    raise NotImplementedError()

