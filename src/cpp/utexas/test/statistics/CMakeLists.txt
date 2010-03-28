# the test suite
find_package(
    Boost REQUIRED COMPONENTS
    unit_test_framework
    )

add_library(
    ut_statistics_tests SHARED
    multinomial.cc
    )

target_link_libraries(
    ut_statistics_tests
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

