#
# build
#

add_cython_module(cargo_llvm__high_level _high_level)

#
# install
#

install(
    TARGETS
        cargo_llvm__high_level
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

