# locate Python
include(FindPythonLibs)
include(FindPythonInterp)

# locate numpy
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/echo_numpy_path.py"
    OUTPUT_VARIABLE NUMPY_FOUND_PATH
    ERROR_VARIABLE NUMPY_FOUND_PATH
    RESULT_VARIABLE ECHO_NUMPY_PATH_FAILED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

if(ECHO_NUMPY_PATH_FAILED)
    message(SEND_ERROR "Path to numpy could not be found: ${NUMPY_FOUND_PATH}")
endif(ECHO_NUMPY_PATH_FAILED)

message(STATUS "Path to numpy headers guessed as: ${NUMPY_FOUND_PATH}")

# locate Boost.Python
find_package(
    Boost REQUIRED COMPONENTS
    python
    )

set(
    UT_PYTHON_INCLUDE_DIRS
    "${NUMPY_FOUND_PATH}/core/include"
    ${PYTHON_INCLUDE_DIRS}
    CACHE INTERNAL "Header search paths for utexas/python."
    )

include_directories(
    ${UT_PYTHON_INCLUDE_DIRS}
    )

# set up the actual build targets
add_library(
    ut_python SHARED
    _statistics.cc
    numpy.cc
    numpy_array.cc
    numpy_symbol.cc
    randomness.cc
    )

# FIXME no idea what the statistics-python dependency graph should look like
target_link_libraries(
    ut_python
	gsl
	gslcblas
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )

set_target_properties(ut_python PROPERTIES OUTPUT_NAME _statistics PREFIX "")

install(TARGETS ut_python DESTINATION cargo/statistics)
install(
    FILES
    numpy.h
    randomness.h
    numpy_array.h
    numpy_symbol.h
    get_shared_ptr.h
    DESTINATION include/utexas/python
    )

