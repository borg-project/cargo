"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import assert_equal

def test_chunk_divisible():
    """
    Test chunking of a nicely-divisible sequence.
    """

    from cargo.iterators import chunk

    sequence = range(20)
    chunks   = [list(s) for s in chunk(sequence, 4)]

    assert_equal(chunks[0], range( 0,  4))
    assert_equal(chunks[1], range( 4,  8))
    assert_equal(chunks[2], range( 8, 12))
    assert_equal(chunks[3], range(12, 16))
    assert_equal(chunks[4], range(16, 20))

def test_chunk_not_divisible():
    """
    Test chunking of a sequence not nicely divisible.
    """

    from cargo.iterators import chunk

    sequence = range(18)
    chunks   = [list(s) for s in chunk(sequence, 4)]

    assert_equal(chunks[0], range( 0,  4))
    assert_equal(chunks[1], range( 4,  8))
    assert_equal(chunks[2], range( 8, 12))
    assert_equal(chunks[3], range(12, 16))
    assert_equal(chunks[4], range(16, 18))

