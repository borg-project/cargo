"""
cargo/kit/bases.py

General support base classes.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc import ABCMeta

class ABC(object):
    """
    Base class for abstract base classes.

    Completely unecessary, but makes ABCs slightly more convenient.
    """

    __metaclass__ = ABCMeta

