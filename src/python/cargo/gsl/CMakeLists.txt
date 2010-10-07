#
# build
#

add_cython_module(cargo_gsl_sf      sf)
add_cython_module(cargo_gsl_errno   errno)
add_cython_module(cargo_gsl_randist randist)

target_link_libraries(cargo_gsl_sf      ${GSL_LIBRARIES})
target_link_libraries(cargo_gsl_errno   ${GSL_LIBRARIES})
target_link_libraries(cargo_gsl_randist ${GSL_LIBRARIES})

#
# install
#

install(
    TARGETS
        cargo_gsl_sf
        cargo_gsl_errno
        cargo_gsl_randist
    DESTINATION lib/cargo/gsl
    )
install(
    FILES
        __init__.py
    DESTINATION lib/cargo/gsl
    )

