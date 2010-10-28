"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from nose.tools import (
    assert_true,
    assert_equal,
    )

def test_normalize_dtype():
    """
    Test normalization of numpy dtypes.
    """

    from cargo.numpy import normalize_dtype

    weird  = numpy.dtype(((int, (2,)), (4,)))
    normal = normalize_dtype(weird)

    assert_equal(normal.base, int)
    assert_equal(normal.shape, (4, 2))
    assert_equal(normal, weird)

def test_semicast_full():
    """
    Test full ndarray broadcasting via semicast.
    """

    from cargo.numpy import semicast

    a = numpy.arange(2)
    b = numpy.arange(4)

    (shape, (aa, bb)) = semicast((a, None), (b[:, None], None))

    assert_equal(shape, (4, 2))
    assert_equal(aa.shape, (4, 2))
    assert_equal(bb.shape, (4, 2))
    assert_equal(aa.tolist(), [[0, 1]] * 4)
    assert_equal(bb.tolist(), [[0, 0], [1, 1], [2, 2], [3, 3]])

def test_semicast_full_fields():
    """
    Test full ndarray broadcasting via semicast, with struct dtype.
    """

    from cargo.numpy import semicast

    d = numpy.dtype([("v", numpy.uint)])
    a = numpy.array([(i,) for i in xrange(2)], d)
    b = numpy.array([(i,) for i in xrange(4)], d)

    (shape, (aa, bb)) = semicast((a, None), (b[:, None], None))

    assert_equal(shape, (4, 2))
    assert_equal(aa.shape, (4, 2))
    assert_equal(bb.shape, (4, 2))
    assert_equal(aa.dtype, d)
    assert_equal(bb.dtype, d)
    assert_equal(aa.tolist(), [[(0,), (1,)]] * 4)
    assert_equal(bb.tolist(), [[(i,), (i,)] for i in xrange(4)])

def test_semicast_partial():
    """
    Test partial ndarray broadcasting via semicast.
    """

    from cargo.numpy import semicast

    a = numpy.ones((2, 3, 4))
    b = numpy.zeros((3, 5))

    (shape, (aa, bb)) = semicast((a, -1), (b, -1))

    assert_equal(shape, (2, 3))
    assert_equal(aa.shape, (2, 3, 4))
    assert_equal(bb.shape, (2, 3, 5))
    assert_equal(aa.tolist(), numpy.ones((2, 3, 4)).tolist())
    assert_equal(bb.tolist(), numpy.zeros((2, 3, 5)).tolist())

