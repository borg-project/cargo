# the test suite
find_package(
    Boost REQUIRED COMPONENTS
    unit_test_framework
    )

include_directories(
    ${UT_STATISTICS_INCLUDE_DIRS}
    )

add_library(
    ut_statistics_tests SHARED
    multinomial.cc
    )

target_link_libraries(
    ut_statistics_tests
    ut_statistics
    ${Boost_LIBRARIES}
    )

# the test runner
add_executable(
    ut_test_statistics
    ut_test_statistics.cc
    )

target_link_libraries(
    ut_test_statistics
    ut_statistics_tests
    )

