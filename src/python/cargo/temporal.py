"""
cargo/temporal.py

Code relating to time.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import datetime
import pytz

def utc_now():
    """
    Return a non-naive UTC datetime instance, zoned pytz.utc.
    """

    return pytz.utc.localize(datetime.datetime.utcnow())

