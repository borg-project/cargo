#
# build
#

add_cython_module(cargo_statistics_dcm _dcm)

target_link_libraries(cargo_statistics_dcm ${GSL_LIBRARIES})

#
# install
#

install(
    TARGETS
        cargo_statistics_dcm
    DESTINATION cargo/statistics
    )
install(
    FILES
        __init__.py
        dcm.py
        distribution.py
        functions.py
        mixture.py
        multinomial.py
    DESTINATION cargo/statistics
    )

#
# descend
#

add_subdirectory(test)

