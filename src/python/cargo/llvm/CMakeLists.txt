#
# build
#

add_cython_module(cargo_llvm_high_level high_level)

#
# install
#

install(
    TARGETS
        cargo_llvm_high_level
    DESTINATION lib/cargo/llvm
    )
install(
    FILES
        __init__.py
		constructs.py
		lowloop.py
    DESTINATION lib/cargo/llvm
    )

#
# descend
#

add_subdirectory(test)

