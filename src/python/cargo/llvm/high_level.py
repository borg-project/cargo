"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import llvm.core

from llvm.core  import (
    Type,
    Value,
    Constant,
    )
from cargo.llvm import BuilderStack

class HighLLVM(object):
    """
    Provide a simple higher-level Python-embedded language on top of LLVM.
    """

    def __init__(self):
        """
        Initialize.
        """

    def value(self, value):
        """
        Return a wrapping value.
        """

        return HighValue.from_any(value)

    def type_(self, some_type):
        """
        Return an LLVM type from some kind of type.
        """

        from cargo.llvm import dtype_to_type

        if isinstance(some_type, type):
            return dtype_to_type(numpy.dtype(some_type))
        elif isinstance(some_type, numpy.dtype):
            return dtype_to_type(some_type)
        elif isinstance(some_type, Type):
            return some_type
        else:
            raise TypeError("cannot build type from \"%s\" instance" % type(some_type))

    def for_(self, count):
        """
        Emit a simple for-style loop.
        """

        index_type = Type.int(32)

        if isinstance(count, Value):
            count_value = count
        else:
            count_value = Constant.int(index_type, count)

        def decorator(emit_body):
            """
            Emit the IR for a particular loop body.
            """

            # prepare the loop structure
            builder  = self.builder
            function = builder.basic_block.function

            start = builder.basic_block
            check = function.append_basic_block("for_loop_check")
            flesh = function.append_basic_block("for_loop_flesh")
            leave = function.append_basic_block("for_loop_leave")

            # build the check block
            builder.branch(check)
            builder.position_at_end(check)

            this_index = builder.phi(index_type, "for_loop_index")

            this_index.add_incoming(Constant.int(index_type, 0), start)

            builder.cbranch(
                builder.icmp(
                    llvm.core.ICMP_UGT,
                    count_value,
                    this_index,
                    ),
                flesh,
                leave,
                )

            # build the flesh block
            builder.position_at_end(flesh)

            emit_body(this_index)

            this_index.add_incoming(
                builder.add(this_index, Constant.int(index_type, 1)),
                builder.basic_block,
                )

            builder.branch(check)

            # wrap up the loop
            builder.position_at_end(leave)

        return decorator

    def select(self, boolean, if_true, if_false):
        """
        Conditionally return one of two values.
        """

        return \
            high.value(
                self.builder.select(
                    boolean._value,
                    if_true._value,
                    if_false._value,
                    ),
                )

    def python(self):
        """
        Emit a call to a Python callable.
        """

        from cargo.llvm._high_level import CallPythonDecorator

        return CallPythonDecorator(self.builder)

    def stack_allocate(self, type_):
        """
        Stack-allocate and return a value.
        """

        return HighValue.from_low(self.builder.alloca(type_))

    @property
    def builder(self):
        """
        Return the current IR builder.
        """

        return BuilderStack.local.top()

high = HighLLVM()

class HighValue(object):
    """
    Value in the wrapper language.
    """

    def __init__(self, value):
        """
        Initialize.
        """

        self._value = value

    def __eq__(self, other):
        """
        Return the result of a less-than comparison.
        """

        # XXX support non-floating-point comparisons

        return \
            high.value(
                high.builder.fcmp(
                    llvm.core.FCMP_OEQ,
                    self._value,
                    other._value,
                    ),
                )

    def __ge__(self, other):
        """
        Return the result of a less-than comparison.
        """

        # XXX support non-floating-point comparisons

        return \
            high.value(
                high.builder.fcmp(
                    llvm.core.FCMP_OGE,
                    self._value,
                    other._value,
                    ),
                )

    def __add__(self, other):
        """
        Return the result of an addition.
        """

        # XXX support non-floating-point additions

        return high.value(high.builder.fadd(self._value, other._value))

    def __sub__(self, other):
        """
        Return the result of an addition.
        """

        # XXX support non-floating-point subtractions

        return high.value(high.builder.fsub(self._value, other._value))

    def load(self, name = ""):
        """
        Load the value pointed to by this pointer.
        """

        return \
            HighValue.from_low(
                high.builder.load(self._value, name = name),
                )

    def store(self, pointer):
        """
        Store this value to the specified pointer.
        """

        return high.builder.store(self._value, pointer._value)

    def gep(self, *indices):
        """
        Return a pointer to a component.
        """

        return \
            HighValue.from_low(
                high.builder.gep(
                    self._value,
                    [HighValue.from_any(i)._value for i in indices],
                    ),
                )

    @staticmethod
    def from_any(value):
        """
        Build a high-level wrapped value from some value.
        """

        if isinstance(value, HighValue):
            return value
        elif isinstance(value, Value):
            return HighValue.from_low(value)
        elif isinstance(value, int):
            return \
                HighValue.from_low(
                    Constant.int(
                        Type.int(numpy.dtype(int).itemsize * 8),
                        value,
                        ),
                    )
        elif isinstance(value, long):
            return \
                HighValue.from_low(
                    Constant.int(
                        Type.int(numpy.dtype(long).itemsize * 8),
                        value,
                        ),
                    )
        elif isinstance(value, float):
            return \
                HighValue.from_low(
                    Constant.real(Type.double(), value),
                    )
        else:
            raise TypeError("cannot build value from \"%s\" instance" % type(value))

    @staticmethod
    def from_low(value):
        """
        Build a high-level wrapped value from an LLVM value.
        """

        from llvm.core import Value

        if not isinstance(value, Value):
            raise TypeError("value is not an LLVM value")

        return HighValue(value)

    @property
    def low(self):
        """
        Return the associated LLVM value.
        """

        return self._value

    @property
    def type_(self):
        """
        Return the type of the associated LLVM value.
        """

        return self.type_

class HighPointer(HighValue):
    """
    Pointer value in the wrapper language.
    """

class HighStruct(HighValue):
    """
    Struct value in the wrapper language.
    """

class HighFunction(HighValue):
    """
    Function in the wrapper language.
    """

    def __init__(self, name, return_type, argument_types):
        """
        Initialize.
        """

        module = high.builder.basic_block.function.module

        HighValue.__init__(
            self,
            module.get_or_insert_function(
                Type.function(
                    high.type_(return_type),
                    map(high.type_, argument_types),
                    ),
                name,
                ),
            )

    def __call__(self, *arguments):
        """
        Emit IR for a function call.
        """

        return \
            high.value(
                high.builder.call(
                    self._value,
                    [high.value(v).low for v in arguments],
                    ),
                )

