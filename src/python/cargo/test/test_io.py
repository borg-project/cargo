"""
cargo/test/io.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    eq_,
    timed,
    )

@timed(4)
def test_unxz():
    """
    Test cargo.io.unxz().
    """

    from cargo.io import unxzed

    data = "\xfd7zXZ\x00\x00\x04\xe6\xd6\xb4F\x02\x00!\x01\x16\x00\x00\x00t/\xe5\xa3\x01\x00\x03bar\n\x00}\x9b\x1d)J\x8ck\xc8\x00\x01\x1c\x04o,\x9c\xc1\x1f\xb6\xf3}\x01\x00\x00\x00\x00\x04YZ"

    eq_(unxzed(data), "bar\n")

@timed(4)
def test_xz():
    """
    Test cargo.io.xz().
    """

    from cargo.io import (
        xzed,
        unxzed,
        )

    eq_("foo", unxzed(xzed("foo")))

