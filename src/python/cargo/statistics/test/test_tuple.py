"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from nose.tools       import assert_almost_equal
from cargo.testing    import assert_almost_equal_deep
from cargo.statistics import (
    Tuple,
    Binomial,
    ModelEngine,
    )

#def test_tuple_rv():
    #"""
    #Test random variate generation under the tuple distribution.
    #"""

    #r = RandomState(42)
    #d = TupleDistribution((Discrete([0.1, 0.9]), Discrete([0.9, 0.1])))

    #assert_samples_ok([distribution.random_variate(random) for _ in xrange(4096)])
    #assert_samples_ok(distribution.random_variates(4096, random))

    #assert_almost_equal(float(sum(s[0] for s in samples)) / len(samples), 0.9, places = 2)
    #assert_almost_equal(float(sum(s[1] for s in samples)) / len(samples), 0.1, places = 2)

def test_tuple_ll():
    """
    Test log-likelihood computation under the tuple distribution.
    """

    model  = Tuple([(Binomial(), 2), (Binomial(), 1)])
    engine = ModelEngine(model)

    assert_almost_equal(
        engine.ll(
            ([(0.25, 1), (0.75, 1)], [(0.5, 1)]),
            ([1, 1], [0]),
            ),
        numpy.log(0.25 * 0.75 * 0.5),
        )
    assert_almost_equal(
        engine.ll(
            ([(0.25, 1), (0.75, 1)], [(0.5, 1)]),
            ([0, 1], [0]),
            ),
        numpy.log(0.75 * 0.75 * 0.5),
        )

def test_tuple_ml():
    """
    Test ML parameter estimation under the tuple distribution.
    """

    from cargo.numpy import tolist_deeply

    model  = Tuple([(Binomial(estimation_n = 1), 2), (Binomial(estimation_n = 1), 1)])
    engine = ModelEngine(model)

    assert_almost_equal_deep(
        tolist_deeply(
            engine.ml(
                [([0, 1], [0])] * 2500 + [([1, 0], [1])] * 7500,
                numpy.ones(10000),
                ),
            ),
        ([(0.75, 1), (0.25, 1)], [(0.75, 1)]),
        )
    assert_almost_equal_deep(
        tolist_deeply(
            engine.ml(
                [([0, 1], [0])] * 1000 + [([1, 0], [1])] * 1000,
                [1.00] * 1000 + [3.00] * 1000,
                ),
            ),
        ([(0.75, 1), (0.25, 1)], [(0.75, 1)]),
        )

def test_tuple_map():
    """
    Test MAP parameter estimation under the tuple distribution.
    """

    from cargo.numpy import tolist_deeply

    model  = Tuple([(Binomial(estimation_n = 1), 2), (Binomial(estimation_n = 1), 1)])
    engine = ModelEngine(model)

    assert_almost_equal_deep(
        tolist_deeply(
            engine.map(
                ([(2, 3), (2, 3)], [(2, 3)]),
                [([0, 1], [0])] * 1 + [([1, 0], [1])] * 9,
                numpy.ones(10),
                ),
            ),
        ([(10.0 / 13.0, 1), (2.0 / 13.0, 1)], [(10.0 / 13.0, 1)]),
        )
    assert_almost_equal_deep(
        tolist_deeply(
            engine.map(
                ([(2, 3), (2, 3)], [(2, 3)]),
                [([0, 1], [0])] * 1 + [([1, 0], [1])] * 9,
                [1.00] * 1 + [3.00] * 9,
                ),
            ),
        ([(28.0 / 31.0, 1), (2.0 / 31.0, 1)], [(28.0 / 31.0, 1)]),
        )

