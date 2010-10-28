"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy
import llvm.core

from llvm.core  import (
    Type,
    Value,
    Constant,
    )
from cargo.llvm import (
    iptr_type,
    BuilderStack,
    )

object_type     = Type.struct([])
object_ptr_type = Type.pointer(object_type)

class HighStandard(object):
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

    value_from_any = value

    def type_(self, some_type):
        """
        Return an LLVM type from some kind of type.
        """

        # XXX support for other ctypes

        from ctypes     import sizeof
        from cargo.llvm import dtype_to_type

        ctype_integer_types = \
            set([
                ctypes.c_bool,
                ctypes.c_byte,
                ctypes.c_ubyte,
                ctypes.c_char,
                ctypes.c_wchar,
                ctypes.c_short,
                ctypes.c_ushort,
                ctypes.c_long,
                ctypes.c_longlong,
                ctypes.c_ulong,
                ctypes.c_ulonglong,
                ctypes.c_int8,
                ctypes.c_int16,
                ctypes.c_int32,
                ctypes.c_int64,
                ctypes.c_uint8,
                ctypes.c_uint16,
                ctypes.c_uint32,
                ctypes.c_uint64,
                ctypes.c_size_t,
                ])

        if isinstance(some_type, type):
            return dtype_to_type(numpy.dtype(some_type))
        elif isinstance(some_type, numpy.dtype):
            return dtype_to_type(some_type)
        elif isinstance(some_type, Type):
            return some_type
        elif some_type in ctype_integer_types:
            return Type.int(sizeof(some_type) * 8)
        else:
            raise TypeError("cannot build type from \"%s\" instance" % type(some_type))

    type_from_any = type_

    def for_(self, count):
        """
        Emit a simple for-style loop.
        """

        index_type = Type.int(32)

        count = self.value(count)

        def decorator(emit_body):
            """
            Emit the IR for a particular loop body.
            """

            # prepare the loop structure
            builder  = self.builder
            function = self.function
            start    = self.basic_block
            check    = function.append_basic_block("for_loop_check")
            flesh    = function.append_basic_block("for_loop_flesh")
            leave    = function.append_basic_block("for_loop_leave")

            # build the check block
            builder.branch(check)
            builder.position_at_end(check)

            this_index = builder.phi(index_type, "for_loop_index")

            this_index.add_incoming(Constant.int(index_type, 0), start)

            builder.cbranch(
                builder.icmp(
                    llvm.core.ICMP_UGT,
                    count.low,
                    this_index,
                    ),
                flesh,
                leave,
                )

            # build the flesh block
            builder.position_at_end(flesh)

            emit_body(high.value(this_index))

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
            self.value(
                self.builder.select(
                    self.value(boolean)._value,
                    self.value(if_true)._value,
                    self.value(if_false)._value,
                    ),
                )

    def random(self):
        """
        Emit a PRNG invocation.
        """

        from cargo.llvm.support import emit_random_real_unit

        return emit_random_real_unit(self)

    def random_int(self, upper, width = 32):
        """
        Emit a PRNG invocation.
        """

        from cargo.llvm.support import emit_random_int

        return emit_random_int(self, upper, width)

    def python(self, *arguments):
        """
        Emit a call to a Python callable.
        """

        return CallPythonDecorator(arguments)

    def heap_allocate(self, type_, count = 1):
        """
        Stack-allocate and return a value.
        """

        from cargo.llvm import sizeof_type

        type_  = self.type_from_any(type_)
        malloc = HighFunction("malloc", Type.pointer(Type.int(8)), [long])
        bytes_ = (self.value(count) * sizeof_type(type_)).cast_to(long)

        return malloc(bytes_).cast_to(Type.pointer(type_))

    def stack_allocate(self, type_, initial = None):
        """
        Stack-allocate and return a value.
        """

        allocated = HighValue.from_low(self.builder.alloca(self.type_from_any(type_)))

        if initial is not None:
            self.value(initial).store(allocated)

        return allocated

    @property
    def builder(self):
        """
        Return the current IR builder.
        """

        return BuilderStack.local.top()

    @property
    def basic_block(self):
        """
        Return the current basic block.
        """

        return self.builder.basic_block

    @property
    def function(self):
        """
        Return the current function.
        """

        return self.basic_block.function

    @property
    def module(self):
        """
        Return the current module.
        """

        return self.function.module

high = HighStandard()

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
                    high.value(other)._value,
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
                    high.value(other)._value,
                    ),
                )

    def __add__(self, other):
        """
        Return the result of an addition.
        """

        other    = high.value(other)
        lhs_kind = self._value.type.kind
        rhs_kind = other._value.type.kind

        if lhs_kind != rhs_kind:
            raise TypeError("mismatched arguments (%s != %s)" % (lhs_kind, rhs_kind))
        elif lhs_kind == llvm.core.TYPE_INTEGER:
            low_value = high.builder.add(self._value, other._value)
        elif lhs_kind in (llvm.core.TYPE_DOUBLE, llvm.core.TYPE_FLOAT):
            low_value = high.builder.fadd(self._value, other._value)
        else:
            raise TypeError("unsupported argument types for addition")

        return high.value(low_value)

    def __sub__(self, other):
        """
        Return the result of a subtraction.
        """

        return high.value(high.builder.fdiv(self._value, high.value(other)._value))

        other    = high.value(other)
        lhs_kind = self._value.type.kind
        rhs_kind = other._value.type.kind

        if lhs_kind != rhs_kind:
            raise TypeError("mismatched arguments (%s != %s)" % (lhs_kind, rhs_kind))
        elif lhs_kind == llvm.core.TYPE_INTEGER:
            low_value = high.builder.sub(self._value, other._value)
        elif lhs_kind in (llvm.core.TYPE_DOUBLE, llvm.core.TYPE_FLOAT):
            low_value = high.builder.fsub(self._value, other._value)
        else:
            raise TypeError("unsupported argument types for subtraction")

        return high.value(low_value)

    def __mul__(self, other):
        """
        Return the result of a multiplication.
        """

        other    = high.value(other)
        lhs_kind = self._value.type.kind
        rhs_kind = other._value.type.kind

        if lhs_kind != rhs_kind:
            raise TypeError("mismatched arguments (%s != %s)" % (lhs_kind, rhs_kind))
        elif lhs_kind == llvm.core.TYPE_INTEGER:
            low_value = high.builder.mul(self._value, other._value)
        elif lhs_kind in (llvm.core.TYPE_DOUBLE, llvm.core.TYPE_FLOAT):
            low_value = high.builder.fmul(self._value, other._value)
        else:
            raise TypeError("unsupported argument types for multiplication")

        return high.value(low_value)

    def __div__(self, other):
        """
        Return the result of a division.
        """

        other    = high.value(other)
        lhs_kind = self._value.type.kind
        rhs_kind = other._value.type.kind

        if lhs_kind != rhs_kind:
            raise TypeError("mismatched arguments (%s != %s)" % (lhs_kind, rhs_kind))
        elif lhs_kind == llvm.core.TYPE_INTEGER:
            low_value = high.builder.sdiv(self._value, other._value)
        elif lhs_kind in (llvm.core.TYPE_DOUBLE, llvm.core.TYPE_FLOAT):
            low_value = high.builder.fdiv(self._value, other._value)
        else:
            raise TypeError("unsupported argument types for division")

        return high.value(low_value)

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

    def cast_to(self, type_, name = ""):
        """
        Cast this value to the specified type.
        """

        # XXX support more casts
        # XXX move this complexity into the type hierarchy?
        # XXX cleanly handle signedness somehow (explicit "signed" highvalue?)

        type_     = high.type_from_any(type_)
        low_value = None

        if self.type_.kind == llvm.core.TYPE_INTEGER:
            if type_.kind == llvm.core.TYPE_DOUBLE:
                low_value = high.builder.sitofp(self._value, type_, name)
            if type_.kind == llvm.core.TYPE_INTEGER:
                if self.type_.width < type_.width:
                    low_value = high.builder.sext(self._value, type_, name)
        elif self.type_.kind == llvm.core.TYPE_DOUBLE:
            if type_.kind == llvm.core.TYPE_INTEGER:
                low_value = high.builder.fptosi(self._value, type_, name)
        elif self.type_.kind == llvm.core.TYPE_POINTER:
            if type_.kind == llvm.core.TYPE_POINTER:
                low_value = high.builder.bitcast(self._value, type_, name)

        if low_value is None:
            raise TypeError("don't know how to perform this conversion")
        else:
            return high.value(low_value)

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

        return self._value.type

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
                        int(value),
                        ),
                    )
        elif isinstance(value, long):
            return \
                HighValue.from_low(
                    Constant.int(
                        Type.int(numpy.dtype(long).itemsize * 8),
                        long(value),
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

        # sanity
        if not isinstance(value, Value):
            raise TypeError("value is not an LLVM value")

        # generate an appropriate value type
        if value.type.kind == llvm.core.TYPE_INTEGER:
            return HighIntegerValue(value)
        elif value.type.kind == llvm.core.TYPE_DOUBLE:
            return HighRealValue(value)
        else:
            return HighValue(value)

class HighIntegerValue(HighValue):
    """
    Integer value in the wrapper language.
    """

    def to_python(self):
        """
        Emit conversion of this value to a Python object.
        """

        int_from_long = HighFunction("PyInt_FromLong", object_ptr_type, [ctypes.c_long])

        return int_from_long(self._value)

class HighRealValue(HighValue):
    """
    Integer value in the wrapper language.
    """

    def to_python(self):
        """
        Emit conversion of this value to a Python object.
        """

        float_from_double = HighFunction("PyFloat_FromDouble", object_ptr_type, [float])

        return float_from_double(self._value)

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

    def __init__(self, name, return_type, argument_types, new = False):
        """
        Initialize.
        """

        type_ = \
            Type.function(
                high.type_from_any(return_type),
                map(high.type_from_any, argument_types),
                )

        if isinstance(name, int):
            low_value = Constant.int(iptr_type, name).inttoptr(Type.pointer(type_))
        else:
            if new:
                low_value = high.module.add_function(type_, name)
            else:
                low_value = high.module.get_or_insert_function(type_, name)

        HighValue.__init__(self, low_value)

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

    @property
    def arguments(self):
        """
        Return the function argument values.
        """

        return map(high.value_from_any, self._value.args)

class CallPythonDecorator(object):
    """
    Emit calls to Python in LLVM.
    """

    __whatever = []

    def __init__(self, arguments = ()):
        """
        Initialize.
        """

        self._arguments = arguments

    def __call__(self, callable_):
        """
        Emit IR for a particular callable.
        """

        # XXX instead maintain module-level reference to the callable

        CallPythonDecorator.__whatever += [callable_]

        # convert the arguments to Python objects
        from cargo.llvm import (
            constant_pointer,
            constant_pointer_to,
            )

        call_object = \
            HighFunction(
                "PyObject_CallObject",
                object_ptr_type,
                [object_ptr_type, object_ptr_type],
                )
        dec_ref        = HighFunction("Py_DecRef"  , Type.void()    , [object_ptr_type])
        tuple_new      = HighFunction("PyTuple_New", object_ptr_type, [ctypes.c_int   ])
        tuple_set_item = \
            HighFunction(
                "PyTuple_SetItem",
                ctypes.c_int,
                [object_ptr_type, ctypes.c_size_t, object_ptr_type],
                )

        argument_tuple = tuple_new(len(self._arguments))

        for (i, argument) in enumerate(self._arguments):
            tuple_set_item(argument_tuple, i, argument.to_python())

        call_result = \
            call_object(
                constant_pointer_to(callable_, object_ptr_type),
                #constant_pointer(0, object_ptr_type),
                argument_tuple,
                )

        # XXX decrement arguments, return value, etc.

