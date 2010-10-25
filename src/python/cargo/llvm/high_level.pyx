"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import llvm.core

from llvm.core  import (
    Type,
    Constant,
    )
from cargo.llvm import BuilderStack

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

class HighLLVM(object):
    """
    Provide a simple higher-level Python-embedded language on top of LLVM.
    """

    def __init__(self):
        """
        Initialize.
        """

        self._py_object   = Type.struct([])
        self._py_object_p = Type.pointer(Type.struct([]))

    def for_(self, count):
        """
        Emit a simple for-style loop.
        """

        from llvm.core import Value

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

            builder.branch(check)

            # build the check block
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

    def python(self):
        """
        Emit a call to a Python callable.
        """

        module = BuilderStack.local.top().basic_block.function.module

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

        def decorator(callable_):
            """
            Emit IR for a particular callable.
            """

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

            # FIXME decref

        return decorator

    @property
    def builder(self):
        """
        Return the current IR builder.
        """

        return BuilderStack.local.top()

high = HighLLVM()

