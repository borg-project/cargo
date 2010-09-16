"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

def test_add_log():
    """
    Test logarithmic addition (add_log).
    """

    from nose.tools                 import assert_almost_equal
    from cargo.statistics.functions import add_log

    assert_almost_equal(add_log(1.0, 2.0), numpy.log(numpy.exp(1.0) + numpy.exp(2.0)))

def test_log_plus():
    """
    Test logarithmic addition (log_plus).
    """

    from nose.tools                 import assert_almost_equal
    from cargo.statistics.functions import log_plus

    assert_almost_equal(log_plus(1.0, 2.0), numpy.log(numpy.exp(1.0) + numpy.exp(2.0)))

