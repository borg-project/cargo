set(
    UT_STATISTICS_INCLUDE_DIRS
    ${UT_PYTHON_INCLUDE_DIRS}
    CACHE INTERNAL "Header search paths for utexas/statistics."
    )

include_directories(
    ${UT_STATISTICS_INCLUDE_DIRS}
    )

set(
    UT_STATISTICS_LIBRARY_DIRS
    ""
    CACHE INTERNAL "Library search paths for utexas/statistics."
    )

link_directories(
    ${UT_STATISTICS_LIBRARY_DIRS}
    )

#add_library(
#    ut_statistics SHARED
#    multinomial.h
#    multinomial.cc
#    )

#target_link_libraries(
#    ut_statistics
#    ut_python
#    gsl
#    gslcblas
#    )

