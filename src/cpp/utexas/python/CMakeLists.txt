# locate Python
include(FindPythonLibs)
include(FindPythonInterp)

# locate numpy
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/echo_numpy_path.py"
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
    numpy.h
    numpy.cc
    numpy_array.h
    numpy_array.cc
    numpy_symbol.h
    numpy_symbol.cc
    )

# FIXME no idea what the statistics-python dependency graph should look like
target_link_libraries(
    ut_python
	gsl
	gslcblas
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )

