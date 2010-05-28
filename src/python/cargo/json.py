"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import absolute_import

import json
import types

from os.path  import dirname
from cargo.io import expandpath

def load_json(path):
    """
    Load a JSON file from the specified path.
    """

    import json

    with open(path) as file:
        return json.load(file)

def save_json(data, path):
    """
    Save a JSON file to the specified path.
    """

    import json

    with open(path, "w") as file:
        return json.dump(data, file)

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

def jsonify(value):
    """
    Do "the right thing" to make a value JSON-serializable.

    If it fails, return the value unchanged.
    """

    try:
        coerce = value.__json__
    except AttributeError:
        return value
    else:
        return coerce()

