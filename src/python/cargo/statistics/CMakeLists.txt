#
# build
#

add_cython_module(cargo_statistics_dcm         dcm)
add_cython_module(cargo_statistics_mixture     mixture)
add_cython_module(cargo_statistics_discrete    discrete)
add_cython_module(cargo_statistics_functions   functions)
add_cython_module(cargo_statistics_multinomial multinomial)

target_link_libraries(cargo_statistics_dcm         ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_mixture     ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_discrete    ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_functions   ${GSL_LIBRARIES})
target_link_libraries(cargo_statistics_multinomial ${GSL_LIBRARIES})

#
# install
#

install(
    TARGETS
        cargo_statistics_dcm
        cargo_statistics_mixture
        cargo_statistics_discrete
        cargo_statistics_functions
        cargo_statistics_multinomial
    DESTINATION lib/cargo/statistics
    )
install(
    FILES
        __init__.py
		base.py
		tuple.py
		binomial.py
		constant.py
		lowloop.py
    DESTINATION lib/cargo/statistics
    )

#
# descend
#

add_subdirectory(test)

