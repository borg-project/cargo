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
    Test convenient TimeDelta construction.
    """

    from nose.tools     import assert_almost_equal
    from cargo.temporal import seconds

    assert_almost_equal(seconds(5.2).seconds, 5)
    assert_almost_equal(seconds(5.2).microseconds, 2e5)

