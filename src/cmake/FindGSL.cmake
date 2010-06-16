#
# locate the GSL
#

if(NOT GSL_LIBRARIES OR NOT GSL_INCLUDE_DIRS)
    find_path(GSL_INCLUDE_DIR gsl/gsl_math.h)
    find_library(GSL_LIBRARY gsl)

    set(GSL_INCLUDE_DIRS ${GSL_INCLUDE_DIR})
    set(GSL_LIBRARIES ${GSL_LIBRARY})
endif(NOT GSL_LIBRARIES OR NOT GSL_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GSL DEFAULT_MSG GSL_LIBRARIES GSL_INCLUDE_DIRS)

