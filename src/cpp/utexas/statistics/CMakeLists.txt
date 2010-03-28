set(
    UT_STATISTICS_INCLUDE_DIRS
    ""
    CACHE STRING "Header directories for utexas/statistics." FORCE
    )

include_directories(
    ${UT_STATISTICS_INCLUDE_DIRS}
    )

set(
    UT_STATISTICS_LIBRARY_DIRS
    ""
    CACHE STRING "Library directories for utexas/statistics." FORCE
    )

link_directories(
    ${UT_STATISTICS_LIBRARY_DIRS}
    )

add_library(
    ut_statistics SHARED
    family.h
    multinomial.h
    )

target_link_libraries(
    ut_statistics
    )

install(
    TARGETS ut_statistics
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

