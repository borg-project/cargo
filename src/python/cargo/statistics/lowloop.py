"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from llvm.core import (
    Type,
    Module,
    Builder,
    Constant,
    )

def array_data(array):
    """
    Get the array data pointer.
    """

    (data, _) = parameters.__array_interface__["data"]
    uintptr_t = Type.int(TargetData.new("target").pointer_size * 8)
    array_t   = strided_array_type(array)

    return Constant.int(uintptr_t, data).inttoptr(Type.pointer(array_t))

def dtype_to_type(dtype):
    """
    Build an LLVM type matching a numpy dtype.
    """

    mapping = {
        'd' : (lambda _ : Type.double()),
        }

    return mapping[dtype.char](dtype)

def strided_array_type(array, axis = 0):
    """
    Build an LLVM type to represent a strided array's structure.
    """

    if axis == array.ndim:
        return (array.dtype.itemsize, dtype_to_type(array.dtype))
    else:
        (subsize, subtype) = strided_array_type(array, axis + 1)

        count   = array.shape[axis]
        stride  = array.strides[axis]
        padding = stride - subsize * count

        return (
            stride,
            Type.struct([
                Type.array(subtype    , count  ),
                Type.array(Type.int(8), padding),
                ]),
            )

class ArrayLoop(object):
    """
    Iterate over strided arrays.
    """

    def __init__(self, function, shape, exit, arrays, name = "loop"):
        """
        Initialize.
        """

        self._function = function
        self._shape    = shape
        self._arrays   = arrays
        self._name     = name

        data_locations = [Constant.null(Type.pointer(Type.int(8))) for _ in arrays]

        (self._entry, _) = self._loop_over(function, 0, exit, {})

    def _loop_over(self, function, axis, exit, indices):
        """
        Build one level of the array loop.
        """

        # prepare the loop structure
        start = function.append_basic_block("%s_ax%i_start" % (self._name, axis))
        check = function.append_basic_block("%s_ax%i_check" % (self._name, axis))
        flesh = function.append_basic_block("%s_ax%i_flesh" % (self._name, axis))

        start_builder = Builder.new(start)
        check_builder = Builder.new(check)
        flesh_builder = Builder.new(flesh)

        # move into the iteration guard
        start_builder.branch(check)

        # set up the axis index
        size_t     = Type.int(32)
        index      = check_builder.phi(size_t, "i%i" % axis)
        next_index = check_builder.add(index, Constant.int(size_t, 1))

        index.add_incoming(Constant.int(size_t, 0), start)

        # set up the strided locations
        location_t = Type.pointer(Type.int(8))
        locations  = [check_builder.phi(location_t) for a in self._arrays.values()]

        for (location, outer_location) in zip(locations, outer_locations):
            location.add_incoming(outer_location, start)

        strides        = [Constant.int(size_t, a.strides[axis]).inttoptr(location_t) for a in self._arrays.values()]
        next_locations = map(check_builder.add, locations, strides)

        # loop, or not
        from llvm.core import ICMP_UGT

        check_builder.cbranch(
            check_builder.icmp(
                ICMP_UGT,
                Constant.int(size_t, self._shape[axis]),
                index,
                ),
            flesh,
            exit,
            )

        if axis != len(self._shape) - 1:
            # build the inner loop
            (inner_start, inner_check) = self._loop_over(function, axis + 1, check, locations)

            index.add_incoming(next_index, inner_check)

            for (location, next_location) in zip(locations, next_locations):
                location.add_incoming(next_location, inner_check)

            flesh_builder.branch(inner_start)

            return (start, inner_check)
        else:
            # we are the innermost loop
            # FIXME we can move these lines into the common control stream
            index.add_incoming(next_index, flesh)

            for (location, next_location) in zip(locations, next_locations):
                location.add_incoming(next_location, flesh)

            repeat = flesh_builder.branch(check)

            flesh_builder.position_before(repeat)

            return (start, check)

    @property
    def locations(self):
        """
        Named location pointers.
        """

        raise NotImplementedError()

    @property
    def entry(self):
        """
        Entry point basic block.
        """

        return self._entry

def foo():
    shape  = (2, 4)
    arrays = {
        "foo" : numpy.random.randint(10, size = shape),
        "bar" : numpy.random.randint(10, size = shape),
        }

    local = Module.new("local")
    main  = local.add_function(Type.function(Type.void(), []), "main")
    entry = main.append_basic_block("entry")
    exit  = main.append_basic_block("exit")
    loop  = ArrayLoop(main, shape, exit, arrays)

    Builder.new(entry).branch(loop.entry)
    Builder.new(exit).ret_void()

    print local

    from llvm.ee   import (
        TargetData,
        ExecutionEngine,
        )

    engine = ExecutionEngine.new(local)

    engine.run_function(main, [])

if __name__ == "__main__":
    foo()

