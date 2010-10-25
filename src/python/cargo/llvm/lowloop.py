"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import ctypes
import numpy
import llvm.core

from llvm.core import (
    Type,
    Constant,
    )
from cargo.llvm            import iptr_type
from cargo.llvm.high_level import high

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

def get_ndarray_type(array, axis = 0):
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

def get_ndarray_data_pointer(array):
    """
    Get the array data as a typed LLVM pointer.
    """

    (data, _)       = array.__array_interface__["data"]
    (array_type, _) = strided_array_type(array)

    return Constant.int(iptr_type, data).inttoptr(Type.pointer(array_type))

class StridedArrays(object):
    """
    Emit IR for manipulating strided arrays of compatible shape.
    """

    def __init__(self, arrays):
        """
        Initialize.
        """

        self._arrays = dict(arrays)

    @staticmethod
    def from_numpy(ndarrays):
        """
        Build from a dictionary of ndarrays.
        """

        pairs = ((k, StridedArray.from_numpy(v)) for (k, v) in ndarrays.items())

        return StridedArrays(dict(pairs))

    def at_all(self, *indices):
        """
        Emit IR to return subarrays at a particular location.
        """

        return StridedArrays((k, v.at(*indices)) for (k, v) in self._arrays.items())

    def loop_all(self, axes = None):
        """
        Iterate over strided arrays.
        """

        # argument sanity
        shape = None

        for array in self._arrays.values():
            if shape is None:
                if axes is None:
                    axes = len(array.shape)

                shape = array.shape[:axes]
            elif array.shape[:axes] != shape:
                raise ValueError("incompatible array shape")

        def decorator(emit_inner):
            """
            Emit IR for a particular inner loop body.
            """

            # prepare to emit IR
            def emit_for_axis(d, indices):
                """
                Build one level of the array loop.
                """

                if d == axes:
                    emit_inner(self.at_all(*indices))
                else:
                    @high.for_(shape[d])
                    def _(index):
                        emit_for_axis(d + 1, indices + [index])

            emit_for_axis(0, [])

        return decorator

    def load_all(self):
        """
        Emit IR to load each array into an SSA register.
        """

        load_array = lambda (k, v): (k, v.load())

        return dict(map(load_array, self._arrays.items()))

    def store_all(self, rhs):
        """
        Emit IR to copy other arrays into these.
        """

        for (k, v) in self._arrays.items():
            v.store(rhs.arrays[k])

    @property
    def arrays(self):
        """
        Return the inner arrays.
        """

        return self._arrays

    # XXX properties

class StridedArray(object):
    """
    Emit IR for interaction with a strided array.
    """

    def __init__(self, data, type_, shape, strides = None):
        """
        Initialize.
        """

        # XXX support calculation of contiguous-array strides

        self._data    = data
        self._type    = type_
        self._shape   = shape
        self._strides = strides

    @staticmethod
    def from_numpy(ndarray):
        """
        Build an array from a particular numpy array.
        """

        # XXX maintain reference to array in module; decref in destructor

        type_         = dtype_to_type(ndarray.dtype)
        (location, _) = ndarray.__array_interface__["data"]
        data          = Constant.int(iptr_type, location).inttoptr(Type.pointer(type_))

        return \
            StridedArray(
                data,
                type_,
                [Constant.int(Type.int(32), d) for d in ndarray.shape],
                [Constant.int(Type.int(32), v) for v in ndarray.strides],
                )

    @staticmethod
    def on_stack(type_, shape):
        """
        Allocate and return a contiguous stack-allocated array.
        """

        # XXX

    def at(self, *indices):
        """
        Emit IR to retrieve a subarray at a particular location.
        """

        if len(indices) > len(self._shape):
            raise ValueError("too many indices")

        location = high.builder.ptrtoint(self._data, iptr_type)

        for (index, stride) in zip(indices, self._strides):
            if stride > 0:
                location = high.builder.add(location, high.builder.mul(index, stride))

        return \
            StridedArray(
                high.builder.inttoptr(location, Type.pointer(self._type)),
                self._type,
                self._shape[len(indices):],
                self._strides[len(indices):],
                )

    def load(self, name = ""):
        """
        Emit IR to load the array into an SSA register.
        """

        if len(self._shape) > 0:
            raise ValueError("cannot load non-scalar array")

        return high.builder.load(self._data, name = name)

    def store(self, rhs):
        """
        Emit IR to copy another array into this one.
        """

        if len(self._shape) > 0:
            raise NotImplementedError("non-scalar array stores not yet supported")

        from llvm.core import Value

        if isinstance(rhs, Value):
            high.builder.store(rhs, self._data)
        else:
            if rhs.shape != self._shape:
                raise ValueError("shape mismatch")

            high.builder.store(rhs.load(), self._data)

    @property
    def shape(self):
        """
        The shape of this array.
        """

        return self._shape

    @property
    def strides(self):
        """
        The strides of this array.
        """

        return self._strides

    @property
    def pointer(self):
        """
        The pointer associated with a scalar array.
        """

        if len(self._shape) > 0:
            raise ValueError("no unique pointer for non-scalar array")

        return self._data

