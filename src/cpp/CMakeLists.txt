find_package(
    Boost REQUIRED COMPONENTS
    python
    )

include(FindPythonLibs)

get_filename_component(PYTHON_INSTALL ${PYTHON_LIBRARY} PATH)
get_filename_component(PYTHON_SITE_PACKAGES ${PYTHON_INSTALL}/../site-packages ABSOLUTE)

include_directories(
    "${PYTHON_SITE_PACKAGES}/numpy/core/include"
    "${PYTHON_SITE_PACKAGES}/PyUblas-0.93.1-py2.6-linux-i686.egg/include"
    "${PYTHON_INCLUDE_PATH}"
    )

add_definitions(
#    -DPYUBLAS_HAVE_BOOST_BINDINGS
    -DBOOST_UBLAS_NDEBUG
    )

add_library(
    ConversionsCommon SHARED
    ConversionsCommon.h
    ConversionsCommon.cc
    WrappedCallable.h
    WrappedCallable.cc
    PythonError.h
    PythonError.cc
    )

target_link_libraries(
    ConversionsCommon
    ${Boost_LIBRARIES}
    )

install(
    TARGETS ConversionsCommon
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

subdirs(
    std
    gtk
    log
    boost
    utils
    google
    physics
    control
    gobject
    messaging
    messages
    geodesy
    coroutines
    optimization
    )

add_custom_target(
    PythonModules
    DEPENDS
    py_std
    py_gtk
    py_log
    py_boost
    py_google
    py_utils
    py_physics
    py_control
    py_gobject
    py_coroutines
    py_messaging
    py_messages
    py_geodesy
    py_utils_quaternion
    py_optimization_neat
    VERBATIM
    )

