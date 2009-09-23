"""
cargo/sugar.py

Simple sugar.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc import ABCMeta
from datetime import timedelta

class TimeDelta(timedelta):
    """
    Wrap datetime.timedelta with a few convenience methods.
    """

    @property
    def as_s(self):
        """
        Return the equivalent number of seconds, floating-point.
        """

        return self.days * 86400.0 + self.seconds + self.microseconds / 1E6

class ABC(object):
    """
    Base class for abstract base classes.

    Completely unecessary, but makes ABCs slightly more convenient.
    """

    __metaclass__ = ABCMeta

