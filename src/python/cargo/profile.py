"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def call_profiled(call):
    """
    Call, and return (return, statistics).
    """

    # need a place to dump profiling results
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile() as named:
        # build the call wrapper
        returned = [None]

        def wrapped():
            returned[0] = call()

        # profile the computation
        from cProfile import runctx

        profile = runctx("c()", {}, {"c": wrapped}, named.name)

        # extract the results
        from pstats import Stats

        return (returned[0], Stats(named.name))

def print_call_profiled(call):
    """
    Call, print profile statistics, and return value.
    """

    # make the call
    (value, stats) = call_profiled(call)

    # display a report
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

    return value

