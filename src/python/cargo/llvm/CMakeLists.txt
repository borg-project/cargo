#
# build
#

add_cython_module(cargo_llvm_support support)

target_link_libraries(cargo_llvm_support ${GSL_LIBRARIES})

#
# install
#

install(
    TARGETS
        cargo_llvm_support
    DESTINATION lib/cargo/llvm
    )
install(
    FILES
        __init__.py
		constructs.py
		lowloop.py
		high_level.py
    DESTINATION lib/cargo/llvm
    )

#
# descend
#

add_subdirectory(test)

