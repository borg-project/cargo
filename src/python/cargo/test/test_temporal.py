"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_utc_now():
    """
    Test construction of a UTC date and time.
    """

    from time           import sleep
    from nose.tools     import assert_true
    from cargo.temporal import utc_now

    then = utc_now()

    sleep(2.0)

    assert_true(utc_now() > then)

def test_seconds():
    """
    Test timedelta conversion to seconds.
    """

    from datetime       import timedelta
    from nose.tools     import assert_almost_equal
    from cargo.temporal import seconds

    assert_almost_equal(seconds(timedelta(seconds = 5.0)), 5.0)
    assert_almost_equal(seconds(timedelta(seconds = 5.2)), 5.2)

