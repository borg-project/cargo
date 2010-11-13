"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy
import llvm.core
import llvm.passes

from llvm.core  import (
    Type,
    Constant,
    )
from contextlib import contextmanager
from cargo.log  import get_logger

iptr_type = Type.int(ctypes.sizeof(ctypes.c_void_p) * 8)
logger    = get_logger(__name__, level = "WARNING")

def constant_pointer(address, type_):
    """
    Return an LLVM pointer constant from an address.
    """

    return Constant.int(iptr_type, address).inttoptr(type_)

def constant_pointer_to(object_, type_):
    """
    Return an LLVM pointer constant to a Python object.
    """

    # XXX do this without assuming id behavior (using ctypes?)

    return constant_pointer(id(object_), type_)

def emit_and_execute(module_name = "", optimize = True):
    """
    Prepare for, emit, and run some LLVM IR.
    """

    from cargo.llvm import (
        high,
        HighLanguage,
        )

    def decorator(emit):
        """
        Build an LLVM module, then execute it.
        """

        # construct the module
        with HighLanguage().active() as high:
            emit()

            high.return_()

        module = high.module

        logger.debug("verifying LLVM IR")

        module.verify()

        # optimize it
        from llvm.ee            import ExecutionEngine
        from llvm.passes        import PassManager
        from cargo.llvm.support import raise_if_set

        engine = ExecutionEngine.new(module)

        if optimize:
            manager = PassManager.new()

            manager.add(engine.target_data)

            manager.add(llvm.passes.PASS_FUNCTION_INLINING)
            manager.add(llvm.passes.PASS_PROMOTE_MEMORY_TO_REGISTER)
            manager.add(llvm.passes.PASS_CONSTANT_PROPAGATION)
            manager.add(llvm.passes.PASS_INSTRUCTION_COMBINING)
            manager.add(llvm.passes.PASS_IND_VAR_SIMPLIFY)
            manager.add(llvm.passes.PASS_GEP_SPLITTER)
            manager.add(llvm.passes.PASS_LOOP_SIMPLIFY)
            manager.add(llvm.passes.PASS_LICM)
            manager.add(llvm.passes.PASS_LOOP_ROTATE)
            manager.add(llvm.passes.PASS_LOOP_STRENGTH_REDUCE)
            manager.add(llvm.passes.PASS_LOOP_UNROLL)
            manager.add(llvm.passes.PASS_GVN)
            manager.add(llvm.passes.PASS_DEAD_STORE_ELIMINATION)
            manager.add(llvm.passes.PASS_DEAD_CODE_ELIMINATION)
            manager.add(llvm.passes.PASS_CFG_SIMPLIFICATION)

            logger.debug("running optimization passes on LLVM IR")

            manager.run(module)

        # execute it
        logger.debug("JITing and executing optimized LLVM IR:\n%s", str(module))

        engine.run_function(high.main, [])

        raise_if_set()

    return decorator

def type_from_struct_type(dtype):
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
            members  += [type_from_dtype(field_dtype)]
            position += field_dtype.itemsize

    return Type.packed_struct(members)

def type_from_shaped_dtype(base, shape):
    """
    Build an LLVM type matching a shaped numpy dtype.
    """

    if shape:
        return Type.array(type_from_shaped_dtype(base, shape[1:]), shape[0])
    else:
        return type_from_dtype(base)

def type_from_dtype(dtype):
    """
    Build an LLVM type matching a numpy dtype.
    """

    if dtype.shape:
        return type_from_shaped_dtype(dtype.base, dtype.shape)
    elif numpy.issubdtype(dtype, numpy.integer):
        return Type.int(dtype.itemsize * 8)
    elif dtype == numpy.float64:
        return Type.double()
    elif dtype == numpy.float32:
        return Type.float()
    elif dtype.fields:
        return type_from_struct_type(dtype)
    else:
        raise ValueError("could not build an LLVM type for dtype %s" % dtype.descr)

def size_of_type(type_):
    """
    Return the size of an instance of a type, in bytes.
    """

    return dtype_from_type(type_).itemsize

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

def dtype_from_array_type(type_):
    """
    Build a numpy dtype from an LLVM array type.
    """

    from cargo.numpy import normalize_dtype

    raw_dtype = numpy.dtype((dtype_from_type(type_.element), (type_.count,)))

    return normalize_dtype(raw_dtype)

def dtype_from_struct_type(type_):
    """
    Build a numpy dtype from an LLVM struct type.
    """

    fields = [("f%i" % i, dtype_from_type(f)) for (i, f) in enumerate(type_.elements)]

    return numpy.dtype(fields)

def dtype_from_type(type_):
    """
    Build a numpy dtype from an LLVM type.
    """

    mapping = {
        llvm.core.TYPE_FLOAT   : (lambda _ : numpy.dtype(numpy.float32)),
        llvm.core.TYPE_DOUBLE  : (lambda _ : numpy.dtype(numpy.float64)),
        llvm.core.TYPE_INTEGER : dtype_from_integer_type,
        llvm.core.TYPE_STRUCT  : dtype_from_struct_type,
        llvm.core.TYPE_ARRAY   : dtype_from_array_type,
        }

    return mapping[type_.kind](type_)

