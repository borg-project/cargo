cmake_minimum_required(VERSION 2.6)

# interaction with pkg-config
include(FindPkgConfig)

# interaction with boost
include(FindBoost)

set(Boost_USE_MULTITHREAD ON)
set(Boost_ADDITIONAL_VERSIONS
    "1.42.0"
    "1.42"
    )
set(Boost_MIN_VERSION "1.42")

find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# add a "Development" build type
set(CMAKE_C_FLAGS_DEVELOPMENT
    "-Wall -Werror -pthread -ggdb"
    CACHE STRING
    "Flags used by the C compiler during development builds."
    FORCE
    )
set(CMAKE_CXX_FLAGS_DEVELOPMENT
    "${CMAKE_C_FLAGS_DEVELOPMENT} --std=c++0x"
    CACHE STRING
    "Flags used by the C++ compiler during development builds."
    FORCE
    )
set(CMAKE_EXE_LINKER_FLAGS_DEVELOPMENT
    "-Wl,--warn-unresolved-symbols,--warn-once"
    CACHE STRING
    "Flags used for linking binaries during development builds."
    FORCE
    )
set(CMAKE_SHARED_LINKER_FLAGS_DEVELOPMENT
    "-Wl,--warn-unresolved-symbols,--warn-once"
    CACHE STRING
    "Flags used by the shared libraries linker during development builds."
    FORCE
    )
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEVELOPMENT
    CMAKE_C_FLAGS_DEVELOPMENT
    CMAKE_EXE_LINKER_FLAGS_DEVELOPMENT
    CMAKE_SHARED_LINKER_FLAGS_DEVELOPMENT
    )

# set the default build type
set(CMAKE_BUILD_TYPE
    ${CMAKE_BUILD_TYPE}
    CACHE STRING
    "Choose the type of build; options are: None Development Debug Release RelWithDebInfo MinSizeRel."
    FORCE
    )

# generate the project makefiles
add_subdirectory(
    cpp
    )

