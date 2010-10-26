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

