"""
cargo/json.py

General JSON routines.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import absolute_import

import json
import types

from os.path  import dirname
from cargo.io import expandpath

def follows(value, relative = ""):
    """
    Return non-string values; follow string values.

    Relative paths are assumed to be from C{relative}.
    """

    if type(value) not in types.StringTypes:
        return value
    else:
        path = expandpath(value, relative)

        with open(path) as file:
            return follows(json.load(file), dirname(path))

