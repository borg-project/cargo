"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from datetime import tzinfo

class UTC(tzinfo):
    """
    The one true time zone.
    """

    def utcoffset(self, dt):
        """
        Return the offset to UTC.
        """

        from datetime import timedelta

        return timedelta(0)

    def tzname(self, dt):
        """
        Return the time zone name.
        """

        return "UTC"

    def dst(self, dt):
        """
        Return the DST offset.
        """

        from datetime import timedelta

        return timedelta(0)

def utc_now():
    """
    Return a non-naive UTC datetime instance.
    """

    from datetime import datetime as DateTime

    return DateTime.now(UTC())

def seconds(value):
    """
    Return the equivalent number of seconds, floating-point.
    """

    return value.days * 8.64e4 + value.seconds + value.microseconds / 1e6

def parse_timedelta(s):
    """
    Parse a timedelta string.
    """

    from datetime import timedelta

    return timedelta(seconds = float(s))

