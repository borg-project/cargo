"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from llvm.core  import (
    Type,
    Constant,
    )

from cpython.ref cimport PyObject

cdef constant_pointer(void* pointer, type_):
    """
    Return an LLVM pointer constant from an C pointer.
    """

    from cargo.llvm import iptr_type

    return Constant.int(iptr_type, <long>pointer).inttoptr(type_)

def constant_pointer_to(value, type_):
    """
    Return an LLVM pointer constant from an C pointer.
    """

    return constant_pointer(<void*>value, type_)

class CallPythonDecorator(object):
    """
    Emit calls to Python in LLVM.
    """

    def __init__(self, builder):
        """
        Initialize.
        """

        self._builder     = builder
        self._py_object   = Type.struct([])
        self._py_object_p = Type.pointer(Type.struct([]))

    def __call__(self, callable_):
        """
        Emit IR for a particular callable.
        """

        module      = self._builder.basic_block.function.module
        call_object = \
            module.get_or_insert_function(
                Type.function(
                    self._py_object_p,
                    [self._py_object_p, self._py_object_p],
                    ),
                "PyObject_CallObject",
                )
        dec_ref = \
            module.get_or_insert_function(
                Type.function(Type.void(), [self._py_object_p]),
                "Py_DecRef",
                )

        global whatever

        whatever = callable_

        # FIXME maintain module-level reference to this object

        builder = self.builder

        builder.call(
            call_object,
            [
                constant_pointer_to(callable_, self._py_object_p),
                constant_pointer(NULL, self._py_object_p),
                ],
            )

#cdef void put_double(double v):
    #sys.stdout.write("%s" % v)

#cdef void put_int(int32_t v):
    #sys.stdout.write("%s" % v)

#cdef void put_string(char* string):
    #sys.stdout.write(string)

#def emit_print_string(builder, string):
    #from cargo.llvm import iptr_type

    #print_string_t = Type.pointer(Type.function(Type.void(), [Type.pointer(Type.int(8))]))
    #print_string   = Constant.int(iptr_type, <long>&put_string).inttoptr(print_string_t)

    #from llvm.core import GlobalVariable

    #module    = builder.basic_block.function.module
    #cstring   = GlobalVariable.new(module, Type.array(Type.int(8), len(string) + 1), "cstring")
    #cstring_p = builder.gep(cstring, [Constant.int(Type.int(32), 0)] * 2)

    #cstring.initializer = Constant.stringz(string)

    #builder.call(print_string, [cstring_p])

#def emit_print(builder, *values):
    #import ctypes

    #from ctypes     import sizeof
    #from cargo.llvm import iptr_type

    #print_double_t = Type.pointer(Type.function(Type.void(), [Type.double()]))
    #print_int_t    = Type.pointer(Type.function(Type.void(), [Type.int(32)]))

    #print_double = Constant.int(iptr_type, <long>&put_double).inttoptr(print_double_t)
    #print_int    = Constant.int(iptr_type, <long>&put_int).inttoptr(print_int_t)

    #for value in values:
        #if value.type.kind == llvm.core.TYPE_DOUBLE:
            #builder.call(print_double, [value])
        #elif value.type.kind == llvm.core.TYPE_INTEGER:
            #assert value.type.width == 32

            #builder.call(print_double, [value])

        #emit_print_string(builder, " ")

    #emit_print_string(builder, "\n")

