#
# build
#

add_cython_module(cargo_statistics_dcm _dcm)
add_cython_module(cargo_statistics_functions _functions)
add_cython_module(cargo_statistics_multinomial _multinomial)

target_link_libraries(cargo_statistics_dcm ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_functions ${GSL_LIBRARIES})
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
        dcm.py
		discrete.py
		constant.py
		tuple.py
        functions.py
        mixture.py
        multinomial.py
    DESTINATION lib/cargo/statistics
    )

#
# descend
#

add_subdirectory(test)

