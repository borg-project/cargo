"""
cargo/kit/errors.py

General error routines.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys
import traceback

def print_ignored_error(message = "An error was unavoidably ignored:"):
    """
    We're in an exception handler, but can't handle the exception.
    """

    sys.stderr.write("\n%s\n" % message)

    traceback.print_exc()

    sys.stderr.write("\n")

class Raised(object):
    """
    Store the currently-handled exception.

    The current exception must be saved before errors during error handling are
    handled, so that the original exception can be re-raised with its context
    information intact.
    """

    def __init__(self):
        (self.cls, self.value, self.traceback) = sys.exc_info()

    def re_raise(self):
        raise self.cls, self.value, self.traceback

