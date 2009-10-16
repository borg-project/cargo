"""
cargo/sugar.py

Simple sugar.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc import ABCMeta

def run_once(callable):
    """
    Wrap a callable in a stateful call-once function.
    """

    def wrapper(*args, **kwargs):
        """
        Run the outer callable at most once.
        """

        try:
            callable.__ran_once
        except AttributeError:
            callable.__ran_once = True

            return callable(*args, **kwargs)

    return wrapper

class ABC(object):
    """
    Base class for abstract base classes.

    Completely unecessary, but makes ABCs slightly more convenient.
    """

    __metaclass__ = ABCMeta

