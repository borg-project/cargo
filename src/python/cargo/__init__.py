"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.io import *
from cargo.log import (
    get_logger,
    enable_default_logging,
    )
from cargo.random import *
from cargo.profile import *
from cargo.iterators import *

def get_support_path(name):
    """
    Return the absolute path to a support file.
    """

    from os.path import (
        join,
        exists,
        dirname,
        )

    path = join(dirname(__file__), "_files", name)

    if exists(path):
        return path
    else:
        raise RuntimeError("specified support file does not exist")

