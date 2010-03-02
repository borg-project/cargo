"""
cargo/flags.py

General flags routines.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import sys
import numpy
import optparse

from copy import copy
from optparse import (
    Option,
    OptionGroup,
    OptionParser,
    OptionValueError,
    )
from itertools import (
    chain,
    ifilter,
    )

_flag_sets = []

class FlagSetValues(object):
    """
    Container for flag set values.
    """

    pass

class FlagSet(object):
    """
    Set of flags.
    """

    def __init__(
        self,
        title   = "",
        message = "",
        enabled = True,
        flags   = (),
        given   = FlagSetValues(),
        ):
        """
        Initialize.
        """

        self.title   = title
        self.message = message
        self.enabled = enabled
        self.given   = copy(given)
        self.flags   = flags

        for flag in self.flags:
            assert isinstance(flag, Flag)

        _flag_sets.append(self)

    def merged(self, values):
        """
        Return a new, merged value dictionary.
        """

        new = copy(self.given)

        try:
            items = values.iteritems()
        except AttributeError:
            items = values.__dict__.iteritems()

        new.__dict__.update(items)

        return new

class Flags(FlagSet):
    """
    Typical flag set: a title and a set of flags.
    """

    def __init__(self, title, *args):
        """
        Initialize.
        """

        FlagSet.__init__(self, title = title, flags = args)

class Flag(object):
    """
    Description of a flag.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize this flag description.
        """

        # keep our arguments around
        self.__flag_args   = args
        self.__flag_kwargs = kwargs

        # but construct a basic option now
        self.option = ExtendedOption(*args, **kwargs)

    def add_to(self, container):
        """
        Add this flag to the specified parser or group.
        """

        return container.add_option(*self.__flag_args, **self.__flag_kwargs)

    @property
    def has_default(self):
        """
        Does this option have a default value?
        """

        return self.option.default != ("NO", "DEFAULT")

class IntRange(object):
    """
    A reusable integer range class.
    """

    def __init__(self, *args):
        """
        Initialize.
        """

        if len(args) == 1:
            self.__start = 0
            (self.__stop,) = args
            self.__step = 1
        elif len(args) == 2:
            (self.__start, self.__stop) = args
            self.__step = 1
        elif len(args) == 3:
            (self.__start, self.__stop, self.__step) = args
        else:
            raise RuntimeError("unsupported number of arguments")

    def __iter__(self):
        """
        Return an iterator over this range.
        """

        return iter(self.xrange)

    def __str__(self):
        """
        Return a string representation of this range.
        """

        if self.__step == 1:
            if self.__stop == self.__start + 1:
                return "%i" % self.__start
            else:
                return "%i:%i" % (self.__start, self.__stop)
        else:
            return "%i:%i:%i" % (self.__start, self.__stop, self.__step)

    @staticmethod
    def only(i):
        """
        Return a range over only C{i}.
        """

        return IntRange(i, i + 1)

    @staticmethod
    def optparse_check(object, option, value):
        """
        Parse an IntRange command-line option.
        """

        pieces = value.split(":")

        try:
            ints = [int(p) for p in pieces]

            if len(ints) == 1:
                return IntRange.only(ints[0])
            elif len(ints) <= 3:
                return IntRange(*ints)
        except ValueError:
            pass

        raise OptionValueError("option %s got invalid range value \"%s\"" % (option, value))

    # properties
    start  = property(lambda self: self.__start)
    stop   = property(lambda self: self.__stop)
    step   = property(lambda self: self.__step)
    range  = property(lambda self: range(self.__start, self.__stop, self.__step))
    xrange = property(lambda self: xrange(self.__start, self.__stop, self.__step))

class FloatRange(object):
    """
    A reusable float range class.
    """

    __range_re = re.compile("(?P<start>\\d+(\\.\\d+)?)(:(?P<size>\\d+)\\*(?P<step>\\d+(\\.\\d+)?))?")

    def __init__(self, start, size, step):
        """
        Initialize.
        """

        self.start = start
        self.size = size
        self.step = step

    def __iter__(self):
        """
        Return an iterator over this range.
        """

        for i in xrange(self.size):
            yield self.start + i * self.step

    def __str__(self):
        """
        Return a string representation of this range.
        """

        if self.size == 1:
            return "%f" % self.start
        else:
            return "%f:%i*%f" % (self.start, self.size, self.step)

    @staticmethod
    def only(x):
        """
        Return a range over only C{x}.
        """

        return FloatRange(x, 1, 0.0)

    @staticmethod
    def optparse_check(object, option, value):
        """
        Parse an FloatRange command-line option.
        """

        m = FloatRange.__range_re.match(value)

        if m:
            groups      = m.groupdict()
            start       = float(groups["start"])
            size_string = groups["size"]

            if size_string is None:
                size = 1
                step = 0.0
            else:
                size = int(size_string)
                step = float(groups["step"])

            return FloatRange(start, size, step)
        else:
            raise OptionValueError("option %s got invalid range value \"%s\"" % (option, value))

class IntRanges(object):
    """
    A reusable integer multi-range class.
    """

    def __init__(self, *args):
        """
        Initialize.
        """

        self.__ranges = list(args)

    def __iter__(self):
        """
        Return an iterator over this range.
        """

        return chain(*self.__ranges)

    def __str__(self):
        """
        Return a string representation of this range.
        """

        return ",".join(str(r) for r in self.__ranges)

    @staticmethod
    def only(i):
        """
        Return a range over only C{i}.
        """

        return IntRanges(IntRange.only(i))

    @staticmethod
    def optparse_check(object, option, value):
        """
        Parse a list of IntRange command-line options.
        """

        pieces = value.split(",")

        try:
            return IntRanges(*[IntRange.optparse_check(object, option, p) for p in pieces])
        except ValueError:
            pass

        raise OptionValueError("option %s got invalid ranges value \"%s\"" % (option, value))

    # properties
    ranges = property(lambda self: self.__ranges)

class FloatRanges(object):
    """
    A reusable float multi-range class.
    """

    def __init__(self, *args):
        """
        Initialize.
        """

        self.ranges = list(args)

    def __iter__(self):
        """
        Return an iterator over this range.
        """

        return chain(*self.ranges)

    def __str__(self):
        """
        Return a string representation of this range.
        """

        return ",".join(str(r) for r in self.ranges)

    @staticmethod
    def only(x):
        """
        Return a range over only C{x}.
        """

        return FloatRanges(FloatRange.only(x))

    @staticmethod
    def optparse_check(object, option, value):
        """
        Parse a list of FloatRange command-line options.
        """

        pieces = value.split(",")

        try:
            return FloatRanges(*[FloatRange.optparse_check(object, option, p) for p in pieces])
        except ValueError:
            pass

        raise OptionValueError("option %s got invalid ranges value \"%s\"" % (option, value))

def optparse_check_time_spec(object, option, value):
    """
    Parse a time value.
    """

    try:
        return TimeSpec(ns = float(value) * TimeSpec.NS_PER_S)
    except ValueError:
        pass

    raise OptionValueError("option %s got invalid time value \"%s\"" % (option, value))

class ExtendedOption(Option):
    """
    An optparse option class with support for some useful option types.
    """

    TYPES                       = Option.TYPES + ("IntRange", "IntRanges", "FloatRange", "FloatRanges", "TimeSpec")
    TYPE_CHECKER                = dict(Option.TYPE_CHECKER)
    TYPE_CHECKER["IntRange"]    = IntRange.optparse_check
    TYPE_CHECKER["IntRanges"]   = IntRanges.optparse_check
    TYPE_CHECKER["FloatRange"]  = FloatRange.optparse_check
    TYPE_CHECKER["FloatRanges"] = FloatRanges.optparse_check
    TYPE_CHECKER["TimeSpec"]    = optparse_check_time_spec

    @staticmethod
    def get_parser(usage = None):
        """
        Return a new parser using this option class.
        """

        return OptionParser(option_class = ExtendedOption, usage = usage)

def parse_given(
    argv        = sys.argv,
    extra       = [],
    enable      = set(),
    disable     = set(),
    npositional = None,
    usage       = None,
    ):
    """
    Parse the given flags.
    """

    def is_enabled(s):
        """
        Collect our flags, constructing a parser.
        """

        return (s.enabled and s not in disable) or s in enable

    parser  = ExtendedOption.get_parser(usage = usage)
    origins = {}

    for flag_set in ifilter(is_enabled, _flag_sets):
        values = flag_set.given
        group  = OptionGroup(parser, flag_set.title, flag_set.message)

        for flag in flag_set.flags:
            option = flag.add_to(group)

            assert option.dest not in origins
            assert option.dest not in values.__dict__
            assert option.dest == flag.option.dest

            origins[option.dest] = flag_set

            if flag.has_default:
                # FIXME use OptionGroup.defaults instead
                values.__dict__[option.dest] = option.default

        if group.option_list:
            parser.add_option_group(group)

    # parse given
    (nominal, positional) = arguments = parser.parse_args(argv + extra)

    if npositional and len(positional) < npositional:
        raise RuntimeError("too few positional arguments")

    # store flag values
    for (dest, flag_set) in origins.iteritems():
        flag_set.given.__dict__[dest] = nominal.__dict__[dest]

    # done
    return positional

def with_flags_parsed(*args, **kwargs):
    """
    Return a flag-parsing decorator.
    """

    def decorator(inner_main):
        """
        Return a decorated main.
        """

        def main():
            """
            Body of the decorated main.
            """

            positional = parse_given(*args, **kwargs)

            return inner_main(positional)

        return main

    return decorator

