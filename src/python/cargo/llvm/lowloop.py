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
from cargo.llvm.high_level import (
    high,
    HighValue,
    )

class StridedArrays(object):
    """
    Emit IR for manipulating strided arrays of compatible shape.
    """

    def __init__(self, arrays):
        """
        Initialize.
        """

        self._arrays = dict(arrays)

    def at_all(self, *indices):
        """
        Emit IR to return subarrays at a particular location.
        """

        return StridedArrays((k, v.at(*indices)) for (k, v) in self._arrays.items())

    def loop_all(self, axes = None):
        """
        Iterate over strided arrays.
        """

        print "looping over...", axes

        # argument sanity
        shape = None

        for array in self._arrays.values():
            if shape is None:
                if axes is None:
                    axes = len(array.shape)

                shape = array.shape[:axes]
            #elif array.shape[:axes] != shape:
                #raise ValueError("incompatible array shape")

        print "shape", shape

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

    @staticmethod
    def from_numpy(ndarrays):
        """
        Build from a dictionary of ndarrays.
        """

        pairs = ((k, StridedArray.from_numpy(v)) for (k, v) in ndarrays.items())

        return StridedArrays(dict(pairs))

def get_strided_type(element_type, shape, strides):
    """
    Build an LLVM type to represent a strided array's structure.
    """

    if shape:
        (inner_type, inner_size) = get_strided_type(element_type, shape[1:], strides[1:])

        if strides[0] == 0:
            return (inner_type, inner_size)
        else:
            if strides[0] < inner_size:
                raise ValueError("array stride too small")
            else:
                return (
                    Type.array(
                        Type.packed_struct([
                            inner_type,
                            Type.array(Type.int(8), strides[0] - inner_size),
                            ]),
                        shape[0],
                        ),
                    shape[0] * strides[0],
                    )
    else:
        from cargo.llvm import sizeof_type

        return (element_type, sizeof_type(element_type))

class StridedArray(object):
    """
    Emit IR for interaction with a strided array.
    """

    def __init__(self, strided_data, shape, strides):
        """
        Initialize.
        """

        self._strided_data = strided_data
        self._shape        = shape
        self._strides      = strides

    def at(self, *indices):
        """
        Emit IR to retrieve a subarray at a particular location.
        """

        # sanity
        if len(indices) > len(self._shape):
            raise ValueError("too many indices")

        # build up getelementptr indices
        offsets = []

        for (index, stride) in zip(indices, self._strides):
            if stride > 0:
                offsets += [0, index]

        offsets += [0]

        # return the indexed value
        pointer = self._strided_data.gep(*offsets)

        if len(indices) < len(self._shape):
            return \
                StridedArray(
                    pointer,
                    self._shape[len(indices):],
                    self._strides[len(indices):],
                    )
        else:
            return pointer

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

    @staticmethod
    def from_data_pointer(data, shape, strides = None):
        """
        Build an array from a typical data pointer.

        @param data    : Pointer value (with element-pointer type) to array data.
        @param shape   : Tuple of dimension sizes (Python integers).
        @param strides : Tuple of dimension strides (Python integers).
        """

        shape = map(int, shape)

        if strides is None:
            from cargo.llvm import sizeof_type

            strides   = []
            axis_size = sizeof_type(data.type_.pointee)

            for d in reversed(shape):
                strides   += [axis_size]
                axis_size *= d

            strides = list(reversed(strides))
        else:
            strides = map(int, strides)

        (strided_type, _) = get_strided_type(data.type_.pointee, shape, strides)
        strided_data      = data.cast_to(Type.pointer(strided_type))

        return StridedArray(strided_data, shape, strides)

    @staticmethod
    def heap_allocated(type_, shape):
        """
        Heap-allocate and return a (contiguous) array.
        """

        data = high.heap_allocate(type_, numpy.product(shape))

        return StridedArray.from_data_pointer(data, shape)

    @staticmethod
    def from_numpy(ndarray):
        """
        Build an array from a particular numpy array.
        """

        # XXX maintain reference to array in module; decref in destructor

        from cargo.llvm import dtype_to_type

        type_         = dtype_to_type(ndarray.dtype)
        (location, _) = ndarray.__array_interface__["data"]
        data          = Constant.int(iptr_type, location).inttoptr(Type.pointer(type_))

        return StridedArray.from_data_pointer(high.value(data), ndarray.shape, ndarray.strides)

