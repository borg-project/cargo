"""
cargo/temporal.py

Code relating to time.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import datetime
import pytz

from datetime import timedelta

def utc_now():
    """
    Return a non-naive UTC datetime instance, zoned pytz.utc.
    """

    return pytz.utc.localize(datetime.datetime.utcnow())

class TimeDelta(timedelta):
    """
    Wrap datetime.timedelta with a few convenience methods.
    """

    @staticmethod
    def from_seconds(seconds):
        """
        Return a TimeDelta from a timedelta.
        """

        return TimeDelta(seconds = seconds)

    @staticmethod
    def from_timedelta(delta):
        """
        Return a TimeDelta from a timedelta.
        """

        return \
            TimeDelta(
                days         = delta.days,
                seconds      = delta.seconds,
                microseconds = delta.microseconds,
                )

    @property
    def as_s(self):
        """
        Return the equivalent number of seconds, floating-point.
        """

        return self.days * 86400.0 + self.seconds + self.microseconds / 1E6

