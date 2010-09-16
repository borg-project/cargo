#
# build
#

add_cython_module(cargo_statistics_dcm         dcm)
add_cython_module(cargo_statistics_functions   functions)
add_cython_module(cargo_statistics_multinomial multinomial)

target_link_libraries(cargo_statistics_dcm         ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_functions   ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_multinomial ${GSL_LIBRARIES})

#
# install
#

install(
    TARGETS
        cargo_statistics_dcm
        cargo_statistics_functions
        cargo_statistics_multinomial
    DESTINATION lib/cargo/statistics
    )
install(
    FILES
        __init__.py
		base.py
		discrete.py
		constant.py
		tuple.py
        mixture.py
    DESTINATION lib/cargo/statistics
    )

#
# descend
#

add_subdirectory(test)

