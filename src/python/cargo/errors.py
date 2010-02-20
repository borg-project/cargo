"""
cargo/errors.py

General error routines.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from sys import (
    stderr,
    exc_info,
    )
from traceback import (
    print_exc,
    format_exception,
    )

class Raised(object):
    """
    Store the currently-handled exception.

    The current exception must be saved before errors during error handling are
    handled, so that the original exception can be re-raised with its context
    information intact.
    """

    def __init__(self):
        """
        Initialize.
        """

        (self.type, self.value, self.traceback) = exc_info()

    def format(self):
        """
        Return a list of lines describing the exception.
        """

        return format_exception(self.type, self.value, self.traceback)

    def re_raise(self):
        """
        Re-raise the stored exception.
        """

        raise self.type, self.value, self.traceback

    def print_ignored(self, message = "An error was unavoidably ignored:", file = stderr):
        """
        Print an exception-was-ignored message.
        """

        file.write("\n%s\n" % message)
        file.write("".join(self.format()))
        file.write("\n")

