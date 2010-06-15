"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

class Tests(object):
    """
    Base class for unit tests.
    """

    def setUp(self):
        """
        Test setup.
        """

        self.set_up()

    def tearDown(self):
        """
        Test cleanup.
        """

        self.tear_down()

def assert_almost_equal_deep(left, right, places = 7):
    """
    Assert near-equality, descending into container children.
    """

    from itertools   import izip
    from collections import Container
    from nose.tools  import (
        assert_true,
        assert_false,
        assert_equal,
        assert_almost_equal,
        )

    if isinstance(left, Container):
        assert_true(isinstance(right, Container))
        assert_equal(len(left), len(right))

        for (left_child, right_child) in izip(left, right):
            assert_almost_equal_deep(left_child, right_child, places = places)
    else:
        assert_false(isinstance(right, Container))

        assert_almost_equal(left, right, places = places)

