"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import cargo.defaults

from cargo.io import *
from cargo.log import (
    get_logger,
    enable_default_logging,
    )
from cargo.condor import *
from cargo.labor2 import *
from cargo.sugar import *
from cargo.numpy import *
from cargo.random import *
from cargo.profile import *
from cargo.temporal import *
from cargo.iterators import *
from cargo.testing import *
from cargo.concurrent import *

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

