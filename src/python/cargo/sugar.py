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

class Curried(object):
    """
    A simple pickleable partially-applied callable.
    """

    def __init__(self, callable, *args, **kwargs):
        """
        Initialize.
        """

        self.callable = callable
        self.args     = args
        self.kwargs   = kwargs

    def __call__(self, *args, **kwargs):
        """
        Call the callable.
        """

        keyword = self.kwargs.copy()

        keyword.update(kwargs)

        return self.callable(*(self.args + args), **keyword)

def curry(callable, *args, **kwargs):
    """
    Very simple pickleable partial function application.
    """

    return Curried(callable, *args, **kwargs)

class ABC(object):
    """
    Base class for abstract base classes.

    Completely unecessary, but makes ABCs slightly more convenient.
    """

    __metaclass__ = ABCMeta

