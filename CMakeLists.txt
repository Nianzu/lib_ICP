cmake_minimum_required(VERSION 3.1)
project(lib_icp)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build the test suite" ON)

include_directories(include/)
include_directories(extern/eigen/)

add_subdirectory(src)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()