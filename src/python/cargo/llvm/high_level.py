"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy
import llvm.core

from contextlib import contextmanager
from llvm.core  import (
    Type,
    Value,
    Module,
    Builder,
    Constant,
    Function,
    GlobalVariable,
    )
from cargo.llvm import iptr_type

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

class HighLanguage(object):
    """
    Provide a simple higher-level Python-embedded language on top of LLVM.
    """

    _language_stack = []

    def __init__(self, module = None, test_for_nan = False):
        """
        Initialize.
        """

        # members
        if module is None:
            module = Module.new("high")

        self._module        = module
        self._test_for_nan  = test_for_nan
        self._literals      = {}
        self._builder_stack = []

        # make Python-support declarations
        self._module.add_type_name("PyObjectPtr", Type.pointer(Type.struct([])))

        with self.active():
            # add a main
            main_body = HighFunction.new_named("main_body")

            @HighFunction.define(internal = False)
            def main():
                """
                The true entry point.
                """

                # initialize the Python runtime (matters only for certain test scenarios)
                HighFunction.named("Py_Initialize")()

                # prepare for exception handling
                from cargo.llvm.support import size_of_jmp_buf

                context_type = Type.array(Type.int(8), size_of_jmp_buf())
                context      = GlobalVariable.new(self._module, context_type, "main_context")
                setjmp       = HighFunction.named("setjmp", int, [Type.pointer(Type.int(8))])

                context.linkage     = llvm.core.LINKAGE_INTERNAL
                context.initializer = Constant.null(context_type)

                self.if_(setjmp(context) == 0)(main_body)
                self.return_()

        # prepare for user code
        body_entry = main_body._value.append_basic_block("entry")

        self._builder_stack.append(Builder.new(body_entry))

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

    def string_literal(self, string):
        """
        Define a new string literal.
        """

        if string not in self._literals:
            name  = "literal%i" % len(self._literals)
            value = \
                HighValue.from_low(
                    GlobalVariable.new(self.module, Type.array(Type.int(8), len(string) + 1), name),
                    )

            value._value.linkage     = llvm.core.LINKAGE_INTERNAL
            value._value.initializer = Constant.stringz(string)

            self._literals[string] = value

            return value
        else:
            return self._literals[string]

    def if_(self, condition):
        """
        Emit an if-then statement.
        """

        condition  = self.value_from_any(condition).cast_to(Type.int(1))
        then       = self.function.append_basic_block("then")
        merge      = self.function.append_basic_block("merge")

        def decorator(emit):
            builder = self.builder

            builder.cbranch(condition.low, then, merge)
            builder.position_at_end(then)

            emit()

            if not self.block_terminated:
                builder.branch(merge)

            builder.position_at_end(merge)

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

        if self._test_for_nan:
            self.assert_(~result.is_nan, "result of log(%s) is not a number", value)

        return result

    def log1p(self, value):
        """
        Emit a natural log computation.
        """

        log1p = HighFunction.named("log1p", float, [float])

        log1p._value.add_attribute(llvm.core.ATTR_NO_UNWIND)
        log1p._value.add_attribute(llvm.core.ATTR_READONLY)

        result = log1p(value)

        if self._test_for_nan:
            self.assert_(~result.is_nan, "result of log1p(%s) is not a number", value)

        return result

    def exp(self, value):
        """
        Emit a natural exponentiation.
        """

        exp    = HighFunction.intrinsic(llvm.core.INTR_EXP, [float])
        result = exp(value)

        if self._test_for_nan:
            self.assert_(~result.is_nan, "result of exp(%s) is not a number", value)

        return result

    __whatever = []

    def python(self, *arguments):
        """
        Emit a call to a Python callable.
        """

        def decorator(callable_):
            """
            Emit a call to an arbitrary Python object.
            """

            # XXX properly associate a destructor with the module, etc

            HighLanguage.__whatever += [callable_]

            from cargo.llvm import constant_pointer_to

            HighObject(constant_pointer_to(callable_, self.object_ptr_type))(*arguments)

        return decorator

    def py_import(self, name):
        """
        Import a Python module.
        """

        object_ptr_type = self.module.get_type_named("PyObjectPtr")
        import_         = HighFunction.named("PyImport_ImportModule", object_ptr_type, [Type.pointer(Type.int(8))])

        # XXX error handling

        return HighObject(import_(self.string_literal(name))._value)

    @contextmanager
    def py_scope(self):
        """
        Define a Python object lifetime scope.
        """

        yield HighPyScope()

    def py_tuple(self, *values):
        """
        Build a Python tuple from high-LLVM values.
        """

        tuple_new      = HighFunction.named("PyTuple_New", object_ptr_type, [ctypes.c_int])
        tuple_set_item = \
            HighFunction.named(
                "PyTuple_SetItem",
                ctypes.c_int,
                [object_ptr_type, ctypes.c_size_t, object_ptr_type],
                )

        values_tuple = tuple_new(len(values))

        for (i, value) in enumerate(values):
            if value.type_ == self.object_ptr_type:
                self.py_inc_ref(value)

            tuple_set_item(values_tuple, i, value.to_python())

        return values_tuple

    def py_inc_ref(self, value):
        """
        Decrement the refcount of a Python object.
        """

        inc_ref = HighFunction.named("Py_IncRef", Type.void(), [object_ptr_type])

        inc_ref(value)

    def py_dec_ref(self, value):
        """
        Decrement the refcount of a Python object.
        """

        dec_ref = HighFunction.named("Py_DecRef", Type.void(), [object_ptr_type])

        dec_ref(value)

    def py_print(self, value):
        """
        Print a Python string via sys.stdout.
        """

        if isinstance(value, str):
            value = HighObject.from_string(value)
        elif value.type_ != self.object_ptr_type:
            raise TypeError("py_print() expects a str or object pointer argument")

        with self.py_scope():
            self.py_import("sys").get("stdout").get("write")(value)

    def py_printf(self, format_, *arguments):
        """
        Print arguments via to-Python conversion.
        """

        object_ptr_type = self.object_ptr_type
        py_format       = HighFunction.named("PyString_Format", object_ptr_type, [object_ptr_type] * 2)
        py_from_string  = HighFunction.named("PyString_FromString", object_ptr_type, [Type.pointer(Type.int(8))])

        @HighFunction.define(Type.void(), [a.type_ for a in arguments])
        def py_printf(*inner_arguments):
            """
            Emit the body of the generated print function.
            """

            # build the output string
            format_object    = py_from_string(self.string_literal(format_))
            arguments_object = high.py_tuple(*inner_arguments)
            output_object    = py_format(format_object, arguments_object)

            self.py_dec_ref(format_object)
            self.py_dec_ref(arguments_object)
            self.py_check_null(output_object)

            # write it to the standard output stream
            self.py_print(output_object)
            self.py_dec_ref(output_object)
            self.return_()

        py_printf(*arguments)

    def py_check_null(self, value):
        """
        Bail if a value is null.
        """

        from ctypes import c_int

        @high.if_(value == 0)
        def _():
            longjmp = \
                HighFunction.named(
                    "longjmp",
                    Type.void(),
                    [Type.pointer(Type.int(8)), c_int],
                    )
            context = self.module.get_global_variable_named("main_context")

            longjmp._value.add_attribute(llvm.core.ATTR_NO_RETURN)

            longjmp(context, 1)

    def heap_allocate(self, type_, count = 1):
        """
        Stack-allocate and return a value.
        """

        from cargo.llvm import size_of_type

        type_  = self.type_from_any(type_)
        malloc = HighFunction.named("malloc", Type.pointer(Type.int(8)), [long])
        bytes_ = (self.value_from_any(count) * size_of_type(type_)).cast_to(long)

        return malloc(bytes_).cast_to(Type.pointer(type_))

    def stack_allocate(self, type_, initial = None, name = ""):
        """
        Stack-allocate and return a value.
        """

        allocated = HighValue.from_low(self.builder.alloca(self.type_from_any(type_), name))

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

    def return_(self, value = None):
        """
        Emit a return statement.
        """

        if value is None:
            self.builder.ret_void()
        else:
            self.builder.ret(value._value)

    @contextmanager
    def active(self):
        """
        Make a new language instance active in this context.
        """

        HighLanguage._language_stack.append(self)

        yield self

        HighLanguage._language_stack.pop()

    @contextmanager
    def this_builder(self, builder):
        """
        Temporarily alter the active builder.
        """

        self._builder_stack.append(builder)

        yield builder

        self._builder_stack.pop()

    @property
    def main(self):
        """
        Return the module entry point.
        """

        return self.module.get_function_named("main")

    @property
    def builder(self):
        """
        Return the current IR builder.
        """

        return self._builder_stack[-1]

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

        return self._module

    @property
    def block_terminated(self):
        """
        Does the current basic block end with a terminator?
        """

        return                                                  \
            self.basic_block.instructions                       \
            and self.basic_block.instructions[-1].is_terminator

    @property
    def test_for_nan(self):
        """
        Is NaN testing enabled?
        """

        return self._test_for_nan

    @test_for_nan.setter
    def test_for_nan(self, test_for_nan):
        """
        Is NaN testing enabled?
        """

        self._test_for_nan = test_for_nan

    @property
    def object_ptr_type(self):
        """
        Return the PyObject* type.
        """

        return self.module.get_type_named("PyObjectPtr")

    @staticmethod
    def get_active():
        """
        Get the currently-active language instance.
        """

        return HighLanguage._language_stack[-1]

class HighLanguageDispatcher(object):
    """
    Refer to the currently-active Qy language instance.
    """

    def __getattr__(self, name):
        """
        Retrieve an attribute of the currently-active Qy instance.
        """

        return getattr(HighLanguage.get_active(), name)

high = HighLanguageDispatcher()

class HighValue(object):
    """
    Value in the wrapper language.
    """

    def __init__(self, value):
        """
        Initialize.
        """

        if not isinstance(value, Value):
            raise TypeError("HighValue constructor requires an llvm.core.Value")
        elif self.kind is not None and value.type.kind != self.kind:
            raise TypeError(
                "cannot construct an %s instance from a %s value",
                type(self).__name__,
                type(value).__name,
                )

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

    def __lt__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator < defined" % type(self).__name__)

    def __le__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator <= defined" % type(self).__name__)

    def __gt__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator > defined" % type(self).__name__)

    def __ge__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator >= defined" % type(self).__name__)

    def __eq__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator == defined" % type(self).__name__)

    def __ne__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator != defined" % type(self).__name__)

    def __add__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator + defined" % type(self).__name__)

    def __sub__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator - defined" % type(self).__name__)

    def __mul__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator * defined" % type(self).__name__)

    def __div__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator / defined" % type(self).__name__)

    def __floordiv__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator // defined" % type(self).__name__)

    def __mod__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator % defined" % type(self).__name__)

    def __divmod__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator divmod() defined" % type(self).__name__)

    def __pow__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator ** defined" % type(self).__name__)

    def __and__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator & defined" % type(self).__name__)

    def __xor__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator ^ defined" % type(self).__name__)

    def __or__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator | defined" % type(self).__name__)

    def __lshift__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator << defined" % type(self).__name__)

    def __rshift__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator >> defined" % type(self).__name__)

    def __neg__(self):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have unary operator - defined" % type(self).__name__)

    def __pos__(self):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have unary operator + defined" % type(self).__name__)

    def __abs__(self):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator abs() defined" % type(self).__name__)

    def __invert__(self):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator ~= defined" % type(self).__name__)

    def __iadd__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator += defined" % type(self).__name__)

    def __isub__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator -= defined" % type(self).__name__)

    def __imul__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator *= defined" % type(self).__name__)

    def __idiv__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator /= defined" % type(self).__name__)

    def __ifloordiv__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator //= defined" % type(self).__name__)

    def __imod__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator %= defined" % type(self).__name__)

    def __ipow__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator **= defined" % type(self).__name__)

    def __iand__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator &= defined" % type(self).__name__)

    def __ixor__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator ^= defined" % type(self).__name__)

    def __ior__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator |= defined" % type(self).__name__)

    def __ilshift__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator <<= defined" % type(self).__name__)

    def __irshift__(self, other):
        """
        Explicitly fail for generically-typed values.
        """

        raise TypeError("%s value does not have operator >>= defined" % type(self).__name__)

    def store(self, pointer):
        """
        Store this value to the specified pointer.
        """

        return high.builder.store(self._value, pointer._value)

    @property
    def low(self):
        """
        The associated LLVM value.
        """

        return self._value

    @property
    def type_(self):
        """
        The type of the associated LLVM value.
        """

        return self._value.type

    @property
    def kind(self):
        """
        Enum describing the general kind of this value, or None.
        """

        return None

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
    Failed to coerce a value to that of another type.
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

    def __eq__(self, other):
        """
        Return the result of an equality comparison.
        """

        return \
            HighValue.from_low(
                high.builder.icmp(
                    llvm.core.ICMP_EQ,
                    self._value,
                    high.value_from_any(other)._value,
                    ),
                )

    def __ge__(self, other):
        """
        Return the result of a greater-than comparison.
        """

        return \
            HighValue.from_low(
                high.builder.icmp(
                    llvm.core.ICMP_SGE,
                    self._value,
                    high.value_from_any(other).cast_to(self.type_)._value,
                    ),
                )

    def __le__(self, other):
        """
        Return the result of a less-than comparison.
        """

        return \
            HighValue.from_low(
                high.builder.icmp(
                    llvm.core.ICMP_SLE,
                    self._value,
                    high.value_from_any(other).cast_to(self.type_)._value,
                    ),
                )

    def __add__(self, other):
        """
        Return the result of an addition.
        """

        other = high.value_from_any(other).cast_to(self.type_)

        return HighIntegerValue(high.builder.add(self._value, other._value))

    def __sub__(self, other):
        """
        Return the result of a subtraction.
        """

        other = high.value_from_any(other).cast_to(self.type_)

        return HighIntegerValue(high.builder.sub(self._value, other._value))

    def __mul__(self, other):
        """
        Return the result of a multiplication.
        """

        other = high.value_from_any(other).cast_to(self.type_)

        return HighIntegerValue(high.builder.mul(self._value, other._value))

    def __div__(self, other):
        """
        Return the result of a division.
        """

        other = high.value_from_any(other).cast_to(self.type_)

        return HighIntegerValue(high.builder.sdiv(self._value, other._value))

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
        Return the result of a greater-than comparison.
        """

        return \
            HighValue.from_low(
                high.builder.fcmp(
                    llvm.core.FCMP_OGE,
                    self._value,
                    high.value_from_any(other).cast_to(self.type_)._value,
                    ),
                )

    def __le__(self, other):
        """
        Return the result of a less-than comparison.
        """

        return \
            HighValue.from_low(
                high.builder.fcmp(
                    llvm.core.FCMP_OLE,
                    self._value,
                    high.value_from_any(other).cast_to(self.type_)._value,
                    ),
                )

    def __add__(self, other):
        """
        Return the result of an addition.
        """

        other = high.value_from_any(other).cast_to(self.type_)
        value = HighRealValue(high.builder.fadd(self._value, other._value))

        if high.test_for_nan:
            high.assert_(~value.is_nan, "result of %s + %s is not a number", other, self)

        return value

    def __sub__(self, other):
        """
        Return the result of a subtraction.
        """

        other = high.value_from_any(other).cast_to(self.type_)
        value = HighRealValue(high.builder.fsub(self._value, other._value))

        if high.test_for_nan:
            high.assert_(~value.is_nan, "result of %s - %s is not a number", other, self)

        return value

    def __mul__(self, other):
        """
        Return the result of a multiplication.
        """

        other = high.value_from_any(other).cast_to(self.type_)
        value = HighRealValue(high.builder.fmul(self._value, other._value))

        if high.test_for_nan:
            high.assert_(~value.is_nan, "result of %s * %s is not a number", other, self)

        return value

    def __div__(self, other):
        """
        Return the result of a division.
        """

        other = high.value_from_any(other).cast_to(self.type_)
        value = HighRealValue(high.builder.fdiv(self._value, other._value))

        if high.test_for_nan:
            high.assert_(~value.is_nan, "result of %s / %s is not a number", other, self)

        return value

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

    def to_python(self):
        """
        Build a Python-compatible argument value.
        """

        if self.type_ == high.object_ptr_type:
            return self._value
        else:
            raise TypeError("unknown to-Python conversion for %s" % self.type_)

    def cast_to(self, type_, name = ""):
        """
        Cast this value to the specified type.
        """

        # XXX support more casts

        type_     = high.type_from_any(type_)
        low_value = None

        if type_.kind == llvm.core.TYPE_POINTER:
            low_value = high.builder.bitcast(self._value, type_, name)
        elif type_.kind == llvm.core.TYPE_INTEGER:
            if type_.width == iptr_type.width:
                low_value = high.builder.ptrtoint(self._value, type_, name)

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

        # sanity
        if len(arguments) != len(self.argument_types):
            raise TypeError(
                "function %s expects %i arguments but received %i" % (
                    self._value.name,
                    len(self.argument_types),
                    len(arguments),
                    ),
                )

        # emit the call
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
    def named(name, return_type = Type.void(), argument_types = ()):
        """
        Look up or create a named function.
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
    def new_named(name, return_type = Type.void(), argument_types = (), internal = True):
        """
        Create a named function.
        """

        type_ = \
            Type.function(
                high.type_from_any(return_type),
                map(high.type_from_any, argument_types),
                )
        function = high.module.add_function(type_, name)

        if internal:
            function.linkage = llvm.core.LINKAGE_INTERNAL

        return HighFunction(function)

    @staticmethod
    def define(return_type = Type.void(), argument_types = (), name = None, internal = True):
        """
        Look up or create a named function.
        """

        def decorator(emit):
            """
            Emit the body of the function.
            """

            if name is None:
                if emit.__name__ == "_":
                    function_name = "function"
                else:
                    function_name = emit.__name__
            else:
                function_name = name

            function = HighFunction.new_named(function_name, return_type, argument_types, internal = internal)

            entry = function._value.append_basic_block("entry")

            with high.this_builder(Builder.new(entry)) as builder:
                emit(*function.argument_values)

            return function

        return decorator

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

class HighObject(HighPointerValue):
    """
    Higher-level interface to Python objects in LLVM.
    """

    def __call__(self, *arguments):
        """
        Emit a Python call.
        """

        @HighFunction.define(
            Type.void(),
            [high.object_ptr_type] + [a.type_ for a in arguments],
            )
        def invoke_python(*inner_arguments):
            from cargo.llvm import constant_pointer_to

            call_object = \
                HighFunction.named(
                    "PyObject_CallObject",
                    object_ptr_type,
                    [object_ptr_type, object_ptr_type],
                    )

            argument_tuple = high.py_tuple(*inner_arguments[1:])
            call_result    = call_object(inner_arguments[0], argument_tuple)

            high.py_dec_ref(argument_tuple)
            high.py_check_null(call_result)
            high.py_dec_ref(call_result)
            high.return_()

        invoke_python(self, *arguments)

    def get(self, name):
        """
        Get an attribute.
        """

        object_ptr_type = high.object_ptr_type

        get_attr = HighFunction.named("PyObject_GetAttrString", object_ptr_type, [object_ptr_type, Type.pointer(Type.int(8))])

        result = get_attr(self, high.string_literal(name))

        high.py_check_null(result)

        return HighObject(result._value)

    @staticmethod
    def from_object(instance):
        """
        Build a HighObject for a Python object.
        """

        from cargo.llvm import constant_pointer_to

        return HighObject(constant_pointer_to(instance, high.object_ptr_type))

    @staticmethod
    def from_string(string):
        """
        Build a HighObject for a Python string object.
        """

        py_from_string = HighFunction.named("PyString_FromString", object_ptr_type, [Type.pointer(Type.int(8))])

        return HighObject(py_from_string(high.string_literal(string))._value)

class HighPyScope(object):
    # XXX unimplemented; we're leaking Python objects
    pass

