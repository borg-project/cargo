"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>

Environment variables that matter:
- CARGO_LOG_ROOT_LEVEL
- CARGO_LOG_FILE_PREFIX
"""

# global customization (unfortunate, but whatever)
import logging

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

logging.setLoggerClass(DefaultLogger)

logging.DETAIL = 15
logging.NOTE   = 25

logging.addLevelName(logging.DETAIL, "DETAIL")
logging.addLevelName(logging.NOTE,   "NOTE")

# everything else
from logging     import (
    Filter,
    Logger,
    Formatter,
    FileHandler,
    StreamHandler,
    )

def level_to_number(level):
    """
    Convert a level description to a level number, if necessary.
    """

    if type(level) is str:
        return logging._levelNames[level]
    else:
        return level

def get_logger(name = None, level = None, default_level = logging.WARNING):
    """
    Get or create a logger.
    """

    if name is None:
        logger = logging.root
    else:
        logger = logging.getLogger(name)

    # FIXME the defaults mechanism still isn't quite right

    # set the default level, if the logger is new
    try:
        clean = logger.is_squeaky_clean
    except AttributeError:
        pass
    else:
        if clean and default_level is not None:
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

    _FORMAT      = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _DATE_FORMAT = "%y%m%d%H%M%S"

    def __init__(self):
        """
        Construct this formatter.
        """

        Formatter.__init__(self, VerboseFileFormatter._FORMAT, VerboseFileFormatter._DATE_FORMAT)

def add_console_handler(level = logging.NOTSET, verbose = True):
    """
    Enable typical logging to the console.
    """

    # get the appropriate formatter
    if verbose:
        formatter = TTY_VerboseFormatter
    else:
        formatter = TTY_ConciseFormatter

    # build a handler
    from sys            import stdout
    from cargo.temporal import utc_now

    handler = StreamHandler(stdout)

    handler.setFormatter(formatter(stdout))
    handler.setLevel(level)

    # add it
    logging.root.addHandler(handler)

    log.debug("added log handler for console at %s", utc_now())

    return handler

def add_disk_handler(prefix, level = logging.NOTSET):
    """
    Enable typical logging to disk.
    """

    # generate an unused log file path
    from os.path   import lexists
    from itertools import count

    for i in count():
        path = "%s.%i" % (prefix, i)

        if not lexists(path):
            break

    # build a handler
    from cargo.temporal import utc_now

    handler = FileHandler(path, encoding = "utf-8")

    handler.setFormatter(VerboseFileFormatter())
    handler.setLevel(level)

    # add it
    logging.root.addHandler(handler)

    log.debug("added log handler for file %s at %s", path, utc_now())

    return handler

def enable_default_logging():
    """
    Set up logging in the typical way.
    """

    # configure the default global level
    from cargo import defaults

    get_logger(level = defaults.root_log_level)

    # add the appropriate handlers
    try:
        from os import environ

        prefix = environ["CARGO_LOG_FILE_PREFIX"]
    except KeyError:
        pass
    else:
        add_disk_handler(prefix)

    add_console_handler()

