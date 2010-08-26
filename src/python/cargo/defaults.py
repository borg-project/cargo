"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from os import environ

condor_matching = None
labor_url       = None
root_log_level  = environ.get("CARGO_LOG_ROOT_LEVEL", "NOTSET")

try:
    from cargo_site_defaults import *
except ImportError:
    pass

