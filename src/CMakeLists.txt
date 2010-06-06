project(cargo)

cmake_minimum_required(VERSION 2.8)

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

# required compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -pthread -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pthread -ggdb --std=c++0x")

# generate the project makefiles
add_subdirectory(cpp)

