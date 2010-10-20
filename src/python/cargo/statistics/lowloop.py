"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy
import llvm.core

from llvm.core import (
    Type,
    Builder,
    Constant,
    )

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

def strided_array_type(array, axis = 0):
    """
    Build an LLVM type to represent a strided array's structure.
    """

    if axis == array.ndim:
        return (dtype_to_type(array.dtype), array.dtype.itemsize)
    else:
        (subtype, subsize) = strided_array_type(array, axis + 1)

        stride = array.strides[axis]

        if stride == 0:
            count    = 1
            padding  = 0
            our_size = subsize
        else:
            count    = array.shape[axis]
            padding  = stride - subsize
            our_size = stride * count

        return (
            Type.array(
                Type.packed_struct([
                    subtype,
                    Type.array(Type.int(8), padding),
                    ]),
                count,
                ),
            our_size,
            )

def array_data_pointer(array):
    """
    Get the array data pointer.
    """

    import ctypes

    (data, _)    = array.__array_interface__["data"]
    uintptr_t    = Type.int(ctypes.sizeof(ctypes.c_void_p) * 8)
    (array_t, _) = strided_array_type(array)

    return Constant.int(uintptr_t, data).inttoptr(Type.pointer(array_t))

def strided_array_loop(builder, emit_body, shape, arrays, name = "loop"):
    """
    Iterate over strided arrays.
    """

    assert all(a.shape == shape for a in arrays.values())

    def emit_for_axis(axis, indices):
        """
        Build one level of the array loop.
        """

        # prepare the loop structure
        function = builder.basic_block.function

        start = builder.basic_block
        check = function.append_basic_block("%s_ax%i_check" % (name, axis))
        flesh = function.append_basic_block("%s_ax%i_flesh" % (name, axis))
        leave = function.append_basic_block("%s_ax%i_leave" % (name, axis))

        # set up the axis index
        builder.branch(check)
        builder.position_at_end(check)

        size_type  = Type.int(32)
        zero       = Constant.int(size_type, 0)
        index      = builder.phi(size_type, "%s_ax%i_i" % (name, axis))
        next_index = builder.add(index, Constant.int(size_type, 1))

        index.add_incoming(zero, start)

        # loop conditionally
        builder.cbranch(
            builder.icmp(
                llvm.core.ICMP_UGT,
                Constant.int(size_type, shape[axis]),
                index,
                ),
            flesh,
            leave,
            )
        builder.position_at_end(flesh)

        if axis != len(shape) - 1:
            # build the next-inner loop
            body_value = emit_for_axis(axis + 1, indices + [index])
        else:
            # we are the innermost loop
            locations = {}

            for (array_name, array) in arrays.items():
                offsets = [zero]

                for (i, offset) in enumerate(indices + [index]):
                    if array.strides[i] == 0:
                        offsets += [zero]
                    else:
                        offsets += [offset]

                    offsets += [zero]

                locations[array_name] = \
                    builder.gep(
                        array_data_pointer(array),
                        offsets,
                        "array_%s_loc" % array_name,
                        )

            body_value = emit_body(builder, locations)

        # complete this loop
        index.add_incoming(next_index, builder.basic_block)

        builder.branch(check)
        builder.position_at_end(leave)

        return body_value

    def emit_for_scalar():
        """
        Emit code for a scalar; no loop is required.
        """

        locations = dict((n, array_data_pointer(a)) for (n, a) in arrays.items())

        return emit_body(builder, locations)

    # generate code
    if len(shape) > 0:
        return emit_for_axis(0, [])
    else:
        return emit_for_scalar()

