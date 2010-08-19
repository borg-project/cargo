"""
cargo/log.py

Application-wide logging configuration.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys
import logging

from logging     import (
    Filter,
    Logger,
    Formatter,
    FileHandler,
    StreamHandler,
    )
from cargo.sugar import run_once

class DefaultLogger(logging.getLoggerClass()):
    """
    Simple standard logging.
    """

    def __init__(self, name, level = logging.NOTSET):
        """
        Initialize.
        """

        Logger.__init__(self, name, level)

        self.is_squeaky_clean = True

    def detail(self, message, *args, **kwargs):
        """
        Write a log message at the DETAIL level.
        """

        return self.log(logging.DETAIL, message, *args, **kwargs)

    def note(self, message, *args, **kwargs):
        """
        Write a log message at the NOTE level.
        """

        return self.log(logging.NOTE, message, *args, **kwargs)

# global customization (unfortunate, but whatever)
logging.setLoggerClass(DefaultLogger)

logging.DETAIL = 15
logging.NOTE   = 25

logging.addLevelName(logging.DETAIL, "DETAIL")
logging.addLevelName(logging.NOTE, "NOTE")

def level_to_number(level):
    """
    Convert a level description to a level number, if necessary.
    """

    if type(level) is str:
        return logging._levelNames[level]
    else:
        return level

def get_logger(name, level = None, default_level = logging.WARNING):
    """
    Get or create a logger.
    """

    logger = logging.getLogger(name)

    # FIXME the defaults mechanism still isn't quite right

    # set the default level, if the logger is new
    if logger.is_squeaky_clean:
        if default_level is not None:
            logger.setLevel(level_to_number(default_level))

    # unconditionally set the logger level, if requested
    if level is not None:
        logger.setLevel(level_to_number(level))

        logger.is_squeaky_clean = False

    return logger

log = get_logger(__name__)

class TTY_ConciseFormatter(Formatter):
    """
    A concise log formatter for console output.
    """

    _DATE_FORMAT = "%y%m%d%H%M%S"

    def __init__(self, stream = None):
        """
        Construct this formatter.
        """

        # construct the format string
        format = "%(message)s"

        # initialize this formatter
        Formatter.__init__(self, format, TTY_ConciseFormatter._DATE_FORMAT)

class TTY_VerboseFormatter(Formatter):
    """
    A verbose log formatter for console output.
    """

    _DATE_FORMAT = "%y%m%d%H%M%S"
    _TIME_COLOR  = "\x1b[34m"
    _NAME_COLOR  = "\x1b[35m"
    _LEVEL_COLOR = "\x1b[33m"
    _COLOR_END   = "\x1b[00m"

    def __init__(self, stream = None):
        """
        Construct this formatter.

        Provides colored output if the stream parameter is specified and is an acceptable TTY.
        We print hardwired escape sequences, which will probably break in some circumstances;
        for this unfortunate shortcoming, we apologize.
        """

        # select and construct format string
        import curses

        format = None

        if stream and hasattr(stream, "isatty") and stream.isatty():
            curses.setupterm()

            # FIXME do nice block formatting, increasing column sizes as necessary
            if curses.tigetnum("colors") > 2:
                format = \
                    "%s%%(name)s%s - %s%%(levelname)s%s - %%(message)s" % (
                        TTY_VerboseFormatter._NAME_COLOR,
                        TTY_VerboseFormatter._COLOR_END,
                        TTY_VerboseFormatter._LEVEL_COLOR,
                        TTY_VerboseFormatter._COLOR_END)

        if format is None:
            format = "%(name)s - %(levelname)s - %(message)s"

        # initialize this formatter
        Formatter.__init__(self, format, TTY_VerboseFormatter._DATE_FORMAT)

class VerboseFileFormatter(Formatter):
    """
    A verbose log formatter for file output.
    """

    _FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _DATE_FORMAT = "%y%m%d%H%M%S"

    def __init__(self):
        """
        Construct this formatter.
        """

        Formatter.__init__(self, VerboseFileFormatter._FORMAT, VerboseFileFormatter._DATE_FORMAT)

class OpaqueFilter(Filter):
    """
    Exclude all records.
    """

    def __init__(self):
        """
        Initialize.
        """

        Filter.__init__(self)

    def filter(self, record):
        """
        Allow no records.
        """

        return False

class ExactExcludeFilter(Filter):
    """
    Exclude records with a specific name.
    """

    def __init__(self, exclude):
        """
        Initialize this filter.
        """

        # members
        self.__exclude = exclude

        # base
        Filter.__init__(self)

    def filter(self, record):
        """
        Allow all records save those with the excluded name.
        """

        if record.name != self.__exclude:
            return True
        else:
            return False

def enable_console(level = logging.NOTSET, verbose = True):
    """
    Enable typical logging to the console.
    """

    logging.root.setLevel(logging.NOTSET) # FIXME should use flag

    import datetime

    handler = StreamHandler(sys.stdout)

    if verbose:
        formatter = TTY_VerboseFormatter
    else:
        formatter = TTY_ConciseFormatter

    handler.setFormatter(formatter(sys.stdout))
    handler.setLevel(level)

    logging.root.addHandler(handler)

    log.debug("enabled logging to stdout at %s", datetime.datetime.today().isoformat())

    return handler

enable_console_log = enable_console

def enable_disk(prefix = None, level = logging.NOTSET):
    """
    Enable typical logging to disk.
    """

    # generate an unused log file path
    from os.path   import lexists
    from itertools import count

    if prefix is None:
        from cargo import defaults

        prefix = defaults.log_file_prefix

    for i in count():
        path = "%s.%i" % (prefix, i)

        if not lexists(path):
            break

    # set up logging
    import datetime

    handler = FileHandler(path, encoding = "UTF-8")

    handler.setFormatter(VerboseFileFormatter())
    handler.setLevel(level)

    logging.root.addHandler(handler)

    log.debug("enabled logging to %s at %s", path, datetime.datetime.today().isoformat())

    return handler

@run_once
def enable_default_logging(add_handlers = True):
    # by default, be moderately verbose
    logging.root.setLevel(logging.NOTSET) # FIXME should use flag

    # default setup (FIXME which is silly: should enable the disk log via flag or environment)
    if add_handlers:
        if sys.stdout.isatty():
            enable_console()
        else:
            enable_disk()

