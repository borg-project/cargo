"""
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

def value_by_name(name):
    """
    Look up a value by fully-qualified name.

    Imports modules if necessary.
    """

    parts = name.split(".")
    value = __import__(".".join(parts[:-1]))

    for component in parts[1:]:
        value = getattr(value, component)

    return value

def composed(outer):
    """
    Wrap a callable in a call to outer.
    """

    def decorator(inner):
        from functools import wraps

        @wraps(inner)
        def wrapper(*args, **kwargs):
            return outer(inner(*args, **kwargs))

        return wrapper

    return decorator

def curry(callable, *args, **kwargs):
    """
    Very simple pickleable partial function application.

    We'd use functools.partial if it could be pickled.
    """

    return Curried(callable, *args, **kwargs)

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

class ABC(object):
    """
    Base class for abstract base classes.

    Completely unecessary, but makes ABCs slightly more convenient.
    """

    __metaclass__ = ABCMeta

