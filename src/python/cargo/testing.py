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

