"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import pstats
import tempfile
import cProfile

def call_profiled(call):
    """
    Call, and return (return, statistics).
    """

    with tempfile.NamedTemporaryFile() as named:
        returned = [None]

        def wrapped():
            returned[0] = call()

        profile = cProfile.runctx("c()", {}, {"c": wrapped}, named.name)

        return (returned[0], pstats.Stats(named.name))

def print_call_profiled(call):
    """
    Call, print profile statistics, and return value.
    """

    (value, stats) = call_profiled(call)

    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

    return value

