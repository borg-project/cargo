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
    Function,
    )
from cargo.llvm import (
    iptr_type,
    BuilderStack,
    )

object_type     = Type.struct([])
object_ptr_type = Type.pointer(object_type)

class EmittedAssertionError(AssertionError):
    """
    An assertion was tripped in generated code.
    """

    def __init__(self, message, emission_stack = None):
        """
        Initialize.
        """

        from traceback import extract_stack

        if emission_stack is None:
            emission_stack = extract_stack()[:-1]

        self._emission_stack = emission_stack

        AssertionError.__init__(self, message)

    def __str__(self):
        """
        Generate a human-readable exception message.
        """

        try:
            from traceback import format_list

            return \
                "%s\nCode generation stack:\n%s" % (
                    AssertionError.__str__(self),
                    "".join(format_list(self._emission_stack)),
                    )
        except Exception as error:
            print sys.exc_info()

    @property
    def emission_stack(self):
        """
        Return the stack at the point of assertion IR generation.
        """

        return self._emission_stack

class HighStandard(object):
    """
    Provide a simple higher-level Python-embedded language on top of LLVM.
    """

    def __init__(self, nan_tests = False):
        """
        Initialize.
        """

        self._nan_tests = nan_tests

    def value_from_any(self, value):
        """
        Return a wrapping value.
        """

        return HighValue.from_any(value)

    def type_from_any(self, some_type):
        """
        Return an LLVM type from some kind of type.
        """

        # XXX support for other ctypes

        from ctypes     import sizeof
        from cargo.llvm import type_from_dtype

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
            return type_from_dtype(numpy.dtype(some_type))
        elif isinstance(some_type, numpy.dtype):
            return type_from_dtype(some_type)
        elif isinstance(some_type, Type):
            return some_type
        elif some_type in ctype_integer_types:
            return Type.int(sizeof(some_type) * 8)
        else:
            raise TypeError("cannot build type from \"%s\" instance" % type(some_type))

    def if_(self, condition):
        """
        Emit an if-then statement.
        """

        def decorator(emit_then):
            @self.if_else(condition)
            def _(then):
                if then:
                    emit_then()

        return decorator

    def if_else(self, condition):
        """
        Emit an if-then-else statement.
        """

        condition  = self.value_from_any(condition).cast_to(Type.int(1))
        then       = self.function.append_basic_block("then")
        else_      = self.function.append_basic_block("else")
        merge      = self.function.append_basic_block("merge")

        def decorator(emit_branch):
            builder = self.builder

            builder.cbranch(condition.low, then, else_)
            builder.position_at_end(then)

            emit_branch(True)

            if not self.block_terminated:
                builder.branch(merge)

            builder.position_at_end(else_)

            emit_branch(False)

            if not self.block_terminated:
                builder.branch(merge)

            builder.position_at_end(merge)

        return decorator

    def for_(self, count):
        """
        Emit a simple for-style loop.
        """

        index_type = Type.int(32)

        count = self.value_from_any(count)

        def decorator(emit_body):
            """
            Emit the IR for a particular loop body.
            """

            # prepare the loop structure
            builder  = self.builder
            start    = self.basic_block
            check    = self.function.append_basic_block("for_loop_check")
            flesh    = self.function.append_basic_block("for_loop_flesh")
            leave    = self.function.append_basic_block("for_loop_leave")

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

            emit_body(HighValue.from_low(this_index))

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
            self.value_from_any(
                self.builder.select(
                    self.value_from_any(boolean)._value,
                    self.value_from_any(if_true)._value,
                    self.value_from_any(if_false)._value,
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

    def log(self, value):
        """
        Emit a natural log computation.
        """

        log    = HighFunction.intrinsic(llvm.core.INTR_LOG, [float])
        result = log(value)

        if self._nan_tests:
            self.assert_(~result.is_nan, "result of log(%s) is not a number", value)

        return result

    def log1p(self, value):
        """
        Emit a natural log computation.
        """

        log1p  = HighFunction.named("log1p", float, [float])
        result = log1p(value)

        if self._nan_tests:
            self.assert_(~result.is_nan, "result of log1p(%s) is not a number", value)

        return result

    def exp(self, value):
        """
        Emit a natural exponentiation.
        """

        exp    = HighFunction.intrinsic(llvm.core.INTR_EXP, [float])
        result = exp(value)

        if self._nan_tests:
            self.assert_(~result.is_nan, "result of exp(%s) is not a number", value)

        return result

    def python(self, *arguments):
        """
        Emit a call to a Python callable.
        """

        return CallPythonDecorator(arguments)

    def printf(self, format_, *arguments):
        """
        Emit a call to a Python callable.
        """

        @high.python(*arguments)
        def _(*pythonized):
            print format_ % pythonized

    def heap_allocate(self, type_, count = 1):
        """
        Stack-allocate and return a value.
        """

        from cargo.llvm import sizeof_type

        type_  = self.type_from_any(type_)
        malloc = HighFunction.named("malloc", Type.pointer(Type.int(8)), [long])
        bytes_ = (self.value_from_any(count) * sizeof_type(type_)).cast_to(long)

        return malloc(bytes_).cast_to(Type.pointer(type_))

    def stack_allocate(self, type_, initial = None):
        """
        Stack-allocate and return a value.
        """

        allocated = HighValue.from_low(self.builder.alloca(self.type_from_any(type_)))

        if initial is not None:
            self.value_from_any(initial).store(allocated)

        return allocated

    def assert_(self, boolean, message = "false assertion", *arguments):
        """
        Assert a fact; bails out of the module if false.
        """

        from traceback import extract_stack

        boolean        = self.value_from_any(boolean).cast_to(Type.int(1))
        emission_stack = extract_stack()[:-1]

        @self.if_(~boolean)
        def _():
            # XXX we can do this more simply (avoid the callable argument mangling, etc)
            @self.python(*arguments)
            def _(*pythonized):
                raise EmittedAssertionError(message % pythonized, emission_stack)

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

    @property
    def block_terminated(self):
        """
        Does the current basic block end with a terminator?
        """

        return                                                  \
            self.basic_block.instructions                       \
            and self.basic_block.instructions[-1].is_terminator

high = HighStandard(nan_tests = True)

class HighValue(object):
    """
    Value in the wrapper language.
    """

    def __init__(self, value):
        """
        Initialize.
        """

        self._value = value

    def __str__(self):
        """
        Return a readable string representation of this value.
        """

        return str(self._value)

    def __repr__(self):
        """
        Return a parseable string representation of this value.
        """

        return "HighValue.from_low(%s)" % repr(self._value)

    def __add__(self, other):
        """
        Return the result of an addition.
        """

        other    = high.value_from_any(other)
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

        return HighValue.from_low(low_value)

    def __sub__(self, other):
        """
        Return the result of a subtraction.
        """

        other    = high.value_from_any(other)
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

        return HighValue.from_low(low_value)

    def __mul__(self, other):
        """
        Return the result of a multiplication.
        """

        other    = high.value_from_any(other)
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

        return HighValue.from_low(low_value)

    def __div__(self, other):
        """
        Return the result of a division.
        """

        other    = high.value_from_any(other)
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

        return HighValue.from_low(low_value)

    def store(self, pointer):
        """
        Store this value to the specified pointer.
        """

        return high.builder.store(self._value, pointer._value)

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
        elif isinstance(value, bool):
            return HighValue.from_low(Constant.int(Type.int(1), int(value)))
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
        elif value.type.kind == llvm.core.TYPE_POINTER:
            return HighPointerValue(value)
        else:
            return HighValue(value)

class CoercionError(TypeError):
    """
    Failure to coerce a value to that of another type.
    """

    def __init__(self, from_type, to_type):
        """
        Initialize.
        """

        TypeError.__init__(
            self,
            "don't know how to convert from %s to %s" % (from_type, to_type),
            )

class HighIntegerValue(HighValue):
    """
    Integer value in the wrapper language.
    """

    def __invert__(self):
        """
        Return the result of bitwise inversion.
        """

        return high.builder.xor(self._value, Constant.int(self.type_, -1))

    def cast_to(self, type_, name = ""):
        """
        Cast this value to the specified type.
        """

        # XXX cleanly handle signedness somehow (explicit "signed" highvalue?)

        type_     = high.type_from_any(type_)
        low_value = None

        if type_.kind == llvm.core.TYPE_DOUBLE:
            low_value = high.builder.sitofp(self._value, type_, name)
        elif type_.kind == llvm.core.TYPE_INTEGER:
            if self.type_.width == type_.width:
                low_value = self._value
            elif self.type_.width < type_.width:
                low_value = high.builder.sext(self._value, type_, name)
            elif self.type_.width > type_.width:
                low_value = high.builder.trunc(self._value, type_, name)

        if low_value is None:
            raise CoercionError(self.type_, type_)
        else:
            return HighValue.from_any(low_value)

    def to_python(self):
        """
        Emit conversion of this value to a Python object.
        """

        int_from_long = HighFunction.named("PyInt_FromLong", object_ptr_type, [ctypes.c_long])

        return int_from_long(self._value)

class HighRealValue(HighValue):
    """
    Integer value in the wrapper language.
    """

    def __eq__(self, other):
        """
        Return the result of an equality comparison.
        """

        # XXX support non-floating-point comparisons

        return \
            HighValue.from_low(
                high.builder.fcmp(
                    llvm.core.FCMP_OEQ,
                    self._value,
                    high.value_from_any(other)._value,
                    ),
                )

    def __ge__(self, other):
        """
        Return the result of a less-than comparison.
        """

        # XXX support non-floating-point comparisons

        return \
            HighValue.from_low(
                high.builder.fcmp(
                    llvm.core.FCMP_OGE,
                    self._value,
                    high.value_from_any(other)._value,
                    ),
                )

    @property
    def is_nan(self):
        """
        Test for nan.
        """

        return \
            HighValue.from_low(
                high.builder.fcmp(
                    llvm.core.FCMP_UNO,
                    self._value,
                    self._value,
                    ),
                )

    def cast_to(self, type_, name = ""):
        """
        Cast this value to the specified type.
        """

        # XXX support more casts

        type_     = high.type_from_any(type_)
        low_value = None

        if type_.kind == llvm.core.TYPE_DOUBLE:
            if self.type_.kind == llvm.core.TYPE_DOUBLE:
                low_value = self._value
        if type_.kind == llvm.core.TYPE_INTEGER:
            low_value = high.builder.fptosi(self._value, type_, name)

        if low_value is None:
            raise CoercionError(self.type_, type_)
        else:
            return HighValue.from_low(low_value)

    def to_python(self):
        """
        Emit conversion of this value to a Python object.
        """

        float_from_double = HighFunction.named("PyFloat_FromDouble", object_ptr_type, [float])

        return float_from_double(self._value)

class HighPointerValue(HighValue):
    """
    Pointer value in the wrapper language.
    """

    def __eq__(self, other):
        """
        Return the result of an equality comparison.
        """

        return \
            HighValue.from_low(
                high.builder.icmp(
                    llvm.core.ICMP_EQ,
                    high.builder.ptrtoint(self._value, iptr_type),
                    high.value_from_any(other).cast_to(iptr_type)._value,
                    ),
                )

    def load(self, name = ""):
        """
        Load the value pointed to by this pointer.
        """

        return \
            HighValue.from_low(
                high.builder.load(self._value, name = name),
                )

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

        type_     = high.type_from_any(type_)
        low_value = None

        if type_.kind == llvm.core.TYPE_POINTER:
            low_value = high.builder.bitcast(self._value, type_, name)

        if low_value is None:
            raise CoercionError(self.type_, type_)
        else:
            return HighValue.from_any(low_value)

class HighStructValue(HighValue):
    """
    Struct value in the wrapper language.
    """

class HighFunction(HighValue):
    """
    Function in the wrapper language.
    """

    def __call__(self, *arguments):
        """
        Emit IR for a function call.
        """

        arguments = map(high.value_from_any, arguments)
        coerced   = [v.cast_to(a) for (v, a) in zip(arguments, self.argument_types)]

        return \
            HighValue.from_low(
                high.builder.call(
                    self._value,
                    [c.low for c in coerced],
                    ),
                )

    @property
    def argument_values(self):
        """
        Return the function argument values.

        Meaningful only inside the body of this function.
        """

        return map(high.value_from_any, self._value.args)

    @property
    def argument_types(self):
        """
        Return the function argument values.

        Meaningful only inside the body of this function.
        """

        if self.type_.kind == llvm.core.TYPE_POINTER:
            return self.type_.pointee.args
        else:
            return self.type_.args

    @staticmethod
    def named(name, return_type, argument_types):
        """
        Return a named function.
        """

        type_ = \
            Type.function(
                high.type_from_any(return_type),
                map(high.type_from_any, argument_types),
                )

        return HighFunction(high.module.get_or_insert_function(type_, name))

    @staticmethod
    def get_named(name):
        """
        Look up a named function.
        """

        return HighFunction(high.module.get_function_named(name))

    @staticmethod
    def new_named(name, return_type, argument_types):
        """
        Look up or create a named function.
        """

        type_ = \
            Type.function(
                high.type_from_any(return_type),
                map(high.type_from_any, argument_types),
                )

        return HighFunction(high.module.add_function(type_, name))

    @staticmethod
    def pointed(address, return_type, argument_types):
        """
        Return a function from a function pointer.
        """

        type_ = \
            Type.function(
                high.type_from_any(return_type),
                map(high.type_from_any, argument_types),
                )

        return HighFunction(Constant.int(iptr_type, address).inttoptr(Type.pointer(type_)))

    @staticmethod
    def intrinsic(intrinsic_id, qualifiers = ()):
        """
        Return an intrinsic function.
        """

        qualifiers = map(high.type_from_any, qualifiers)

        return HighFunction(Function.intrinsic(high.module, intrinsic_id, qualifiers))

class IfDecorator(object):
    """
    Emit an if-then-else statement.
    """

    def __init__(self, condition):
        """
        Initialize.
        """

        self._condition = condition

    def __call__(self, emit_then):
        """
        Emit a "then" block.
        """

        return self(emit_then)

    def then(self, emit_then):
        """
        Emit a "then" block.
        """

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

        # build a Python argument tuple
        tuple_new      = HighFunction.named("PyTuple_New", object_ptr_type, [ctypes.c_int])
        tuple_set_item = \
            HighFunction.named(
                "PyTuple_SetItem",
                ctypes.c_int,
                [object_ptr_type, ctypes.c_size_t, object_ptr_type],
                )

        argument_tuple = tuple_new(len(self._arguments))

        for (i, argument) in enumerate(self._arguments):
            tuple_set_item(argument_tuple, i, argument.to_python())

        # call the callable
        from cargo.llvm import constant_pointer_to

        call_object = \
            HighFunction.named(
                "PyObject_CallObject",
                object_ptr_type,
                [object_ptr_type, object_ptr_type],
                )

        call_result = \
            call_object(
                constant_pointer_to(callable_, object_ptr_type),
                argument_tuple,
                )

        # XXX decrement arguments, return value, etc.
        # XXX we could potentially move the bail-on-exception code into a function

        dec_ref = HighFunction.named("Py_DecRef", Type.void(), [object_ptr_type])

        # respond to an exception, if present
        err_occurred = HighFunction.named("PyErr_Occurred", object_ptr_type, [])

        @high.if_else(err_occurred() == 0)
        def _(then):
            if then:
                # XXX emit cleanup code
                pass
            else:
                # XXX can longjmp be marked as noreturn? to avoid the unnecessary br, etc
                longjmp    = HighFunction.intrinsic(llvm.core.INTR_LONGJMP)
                context    = high.module.get_global_variable_named("main_context")
                i8_context = high.builder.bitcast(context, Type.pointer(Type.int(8)))

                longjmp(i8_context, 1)

