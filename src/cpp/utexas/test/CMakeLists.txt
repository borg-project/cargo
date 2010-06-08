# the test runner
add_executable(
    ut_test
    ut_test.cc
    )

target_link_libraries(
    ut_test
#    ut_statistics_tests
    )

# child suites
#add_subdirectory(statistics)

