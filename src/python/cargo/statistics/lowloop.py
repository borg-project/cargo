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

#self._parameter_dtype = \
    #numpy.dtype((
        #[
            #("p", numpy.float_),
            #("c", distribution.parameter_dtype),
            #],
        #K,
        #))

def array_data(array, axis = None):
    """
    Get the array data pointer.
    """

    (data, _) = parameters.__array_interface__["data"]

    uintptr_t = Type.int(TargetData.new("target").pointer_size * 8)

    # FIXME compute the LLVM type for relevant array pointer

    return Constant.int(uintptr_t, data).inttoptr(Type.pointer(array.dtype))

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
        self._exit     = exit
        self._arrays   = arrays
        self._name     = name

        self._entry = self._loop_over(function, 0, exit)

    def _loop_over(self, function, axis, outer):
        """
        Build one level of the array loop.
        """

        from llvm.core import ICMP_UGT

        start = function.append_basic_block("%s_a%i_start" % (self._name, axis))
        check = function.append_basic_block("%s_a%i_check" % (self._name, axis))
        flesh = function.append_basic_block("%s_a%i_flesh" % (self._name, axis))

        start_builder = Builder.new(start)
        check_builder = Builder.new(check)
        flesh_builder = Builder.new(flesh)

        start_builder.branch(check)

        size_t = Type.int(32)
        index  = check_builder.phi(size_t, "i%i" % axis)
        jndex  = check_builder.add(index, Constant.int(size_t, 1))

        index.add_incoming(Constant.int(size_t, 0), start)
        index.add_incoming(jndex                  , flesh)

        check_builder.cbranch(
            check_builder.icmp(
                ICMP_UGT,
                Constant.int(size_t, self._shape[axis]),
                index,
                ),
            flesh,
            outer,
            )

        if axis != len(self._shape) - 1:
            inner = self._loop_over(function, axis + 1, check)

            flesh_builder.branch(inner)
        else:
            from cargo.statistics.mixture import get_zorro

            flesh_builder.call(get_zorro(), [index, index])

            repeat = flesh_builder.branch(check)

            flesh_builder.position_before(repeat)

        return start

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

