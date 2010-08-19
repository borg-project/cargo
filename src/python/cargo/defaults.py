"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

log_file_prefix = "script.log"
condor_matching = None

try:
    from cargo_site_defaults import *
except ImportError:
    pass

