cmake_minimum_required(VERSION 3.1)
project(lib_icp)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS_LIBICP "Build the test suite" ON)

include_directories(include/)
include_directories(extern/eigen/)

# Disable FLANN's tests and examples
set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build FLANN examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Do not build FLANN tests" FORCE)

add_subdirectory(extern/flann/)

add_subdirectory(src)

if(BUILD_TESTS_LIBICP)
    add_subdirectory(test)
endif()