"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes

from llvm.core  import Type
from contextlib import contextmanager

iptr_type = Type.int(ctypes.sizeof(ctypes.c_void_p) * 8)

class BuilderStack(object):
    """
    A stack of IR builders.
    """

    def __init__(self):
        """
        Initialize.
        """

        self._builders = []

    def top(self):
        """
        Return the current IR builder.
        """

        return self._builders[-1]

    def push(self, builder):
        """
        Push an IR builder onto the builder stack.
        """

        self._builders.append(builder)

        return builder

    def pop(self):
        """
        Pop an IR builder off of the builder stack.
        """

        return self._builders.pop()

BuilderStack.local = BuilderStack()

@contextmanager
def this_builder(builder):
    """
    Change the current IR builder for a managed duration.
    """

    BuilderStack.local.push(builder)

    yield builder

    BuilderStack.local.pop()

