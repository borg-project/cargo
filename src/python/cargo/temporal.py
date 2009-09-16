"""
cargo/temporal.py

Code relating to time.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import time
import datetime
import pytz

from datetime import (
    tzinfo,
    timedelta,
    )

class LocalTimeZone(tzinfo):
    """
    The local time zone, using information reported by the time module.

    Note that this implementation is useful only for a datetime instance 
    """

    def utcoffset(self, dt):
        """
        Return the offset of local time from UTC.
        """

        raise NotImplementedError()

#         if time.daylight:
#             return timedelta(seconds = time.altzone)
#         else:
#             return timedelta(seconds = time.timezone)

    def dst(self, dt):
        """
        Return the DST adjustment.
        """

        raise NotImplementedError()

#         if time.daylight:
#             return timedelta(seconds = time.altzone)
#         else:
#             return timedelta()

    def tzname(self, dt):
        """
        Return an arbitrary name for the time zone.
        """

        raise NotImplementedError()

#         (non_dst, dst) = time.tzname

#         if time.daylight:
#             return dst
#         else:
#             return non_dst

def utcnow():
    """
    Return a non-naive UTC datetime instance, zoned pytz.utc.
    """

    return pytz.utc.localize(datetime.datetime.now())

