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

    from llvm.ee import TargetData

    (data, _)    = array.__array_interface__["data"]
    uintptr_t    = Type.int(TargetData.new("target").pointer_size * 8)
    (array_t, _) = strided_array_type(array)

    return Constant.int(uintptr_t, data).inttoptr(Type.pointer(array_t))

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

        count   = array.shape[axis]
        stride  = array.strides[axis]
        padding = stride - subsize

        return (
            Type.array(
                Type.packed_struct([
                    subtype,
                    Type.array(Type.int(8), padding),
                    ]),
                count,
                ),
            stride * count,
            )

class ArrayLoop(object):
    """
    Iterate over strided arrays.
    """

    def __init__(self, function, shape, exit, arrays, name = "loop"):
        """
        Initialize.
        """

        self._function  = function
        self._shape     = shape
        self._arrays    = arrays
        self._name      = name
        self._locations = {}

        (self._entry, _) = self._build_axis_loop(function, 0, exit, [])

    def _build_axis_loop(self, function, axis, exit, indices):
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

        # set up the axis index
        start_builder.branch(check)

        size_t     = Type.int(32)
        zero       = Constant.int(size_t, 0)
        index      = check_builder.phi(size_t, "%s_ax%i_i" % (self._name, axis))
        next_index = check_builder.add(index, Constant.int(size_t, 1))

        index.add_incoming(zero, start)

        # loop conditionally
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
            (inner_start, inner_check) = self._build_axis_loop(function, axis + 1, check, indices + [index])

            index.add_incoming(next_index, inner_check)

            flesh_builder.branch(inner_start)

            return (start, inner_check)
        else:
            # we are the innermost loop
            index.add_incoming(next_index, flesh)

            for (name, array) in self._arrays.items():
                offsets = [zero] + sum(([i, zero] for i in indices + [index]), [])
                gepped  = \
                    flesh_builder.gep(
                        array_data(array),
                        offsets,
                        "%s_lp" % name,
                        )

                self._locations[name] = gepped

            from cargo.statistics.mixture import get_zorro

            #flesh_builder.call(get_zorro(), [flesh_builder.ptrtoint(v, Type.int(64)) for v in self._locations.values()])
            flesh_builder.call(get_zorro(), map(flesh_builder.load, self._locations.values()))
            #flesh_builder.call(get_zorro(), indices + [index])

            repeat = flesh_builder.branch(check)

            flesh_builder.position_before(repeat)

            return (start, check)

    @property
    def locations(self):
        """
        Named location pointers.
        """

        return self._locations

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

    print arrays
    print local

    local.verify()

    from llvm.ee   import (
        TargetData,
        ExecutionEngine,
        )

    engine = ExecutionEngine.new(local)

    engine.run_function(main, [])

if __name__ == "__main__":
    foo()

