"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy

from llvm.core  import (
    Type,
    Constant,
    )
from contextlib import contextmanager

iptr_type = Type.int(ctypes.sizeof(ctypes.c_void_p) * 8)

def constant_pointer(address, type_):
    """
    Return an LLVM pointer constant from an address.
    """

    return Constant.int(iptr_type, address).inttoptr(type_)

def constant_pointer_to(object_, type_):
    """
    Return an LLVM pointer constant to a Python object.
    """

    # XXX do this without calling id (ctypes?)

    return constant_pointer(id(object_), type_)

def emit_and_execute(module_name = ""):
    """
    Prepare for, emit, and run some LLVM IR.
    """

    def decorator(emit):
        # emit some IR
        from llvm.core  import (
            Module,
            Builder,
            )
        from cargo.llvm import this_builder

        module = Module.new(module_name)
        main   = module.add_function(Type.function(Type.void(), []), "main")
        entry  = main.append_basic_block("entry")

        with this_builder(Builder.new(entry)) as builder:
            emit(module)

            builder.ret_void()

        # then compile and execute it
        from llvm.ee import ExecutionEngine

        print module

        module.verify()

        engine = ExecutionEngine.new(module)

        engine.run_function(main, [])

    return decorator

def get_type_size(type_):
    """
    Return the size of an instance of a type, in bytes.
    """

    return dtype_from_type(type_).itemsize

sizeof_type = get_type_size

# XXX type_from_struct_dtype, etc?
def struct_dtype_to_type(dtype):
    """
    Build an LLVM type matching a numpy struct dtype.
    """

    fields   = sorted(dtype.fields.values(), key = lambda (_, p): p)
    members  = []
    position = 0

    for (field_dtype, offset) in fields:
        if offset != position:
            raise NotImplementedError("no support for dtypes with nonstandard packing")
        else:
            members  += [dtype_to_type(field_dtype)]
            position += field_dtype.itemsize

    return Type.packed_struct(members)

def dtype_to_type(dtype):
    """
    Build an LLVM type matching a numpy dtype.
    """

    if numpy.issubdtype(dtype, numpy.integer):
        return Type.int(dtype.itemsize * 8)
    elif dtype == numpy.float64:
        return Type.double()
    elif dtype == numpy.float32:
        return Type.float()
    elif dtype.fields:
        return struct_dtype_to_type(dtype)
    else:
        raise ValueError("could not build an LLVM type for dtype %s" % dtype.descr)

type_from_dtype = dtype_to_type

def dtype_from_integer_type(type_):
    """
    Build a numpy dtype from an LLVM integer type.
    """

    sizes = {
        8  : numpy.dtype(numpy.int8),
        16 : numpy.dtype(numpy.int16),
        32 : numpy.dtype(numpy.int32),
        64 : numpy.dtype(numpy.int64),
        }

    return sizes[type_.width]

def dtype_from_struct_type(type_):
    """
    Build a numpy dtype from an LLVM struct type.
    """

    fields = [
        ("f%i" % i, dtype_from_type(f))
        for (i, f) in enumerate(type_.elements)
        ]

    return numpy.dtype(fields)

def dtype_from_type(type_):
    """
    Build a numpy dtype from an LLVM type.
    """

    from llvm import core

    mapping = {
        core.TYPE_FLOAT   : (lambda _ : numpy.dtype(numpy.float32)),
        core.TYPE_DOUBLE  : (lambda _ : numpy.dtype(numpy.float64)),
        core.TYPE_INTEGER : dtype_from_integer_type,
        core.TYPE_STRUCT  : dtype_from_struct_type,
        }

    return mapping[type_.kind](type_)

class BuilderStack(object):
    """
    A stack of IR builders.
    """

    def __init__(self):
        """
        Initialize.
        """

        self._builders = []

    def top(self):
        """
        Return the current IR builder.
        """

        return self._builders[-1]

    def push(self, builder):
        """
        Push an IR builder onto the builder stack.
        """

        self._builders.append(builder)

        return builder

    def pop(self):
        """
        Pop an IR builder off of the builder stack.
        """

        return self._builders.pop()

BuilderStack.local = BuilderStack()

@contextmanager
def this_builder(builder):
    """
    Change the current IR builder for a managed duration.
    """

    BuilderStack.local.push(builder)

    yield builder

    BuilderStack.local.pop()

