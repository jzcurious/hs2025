cmake_minimum_required(VERSION 3.24)
project(work1-tests LANGUAGES CXX)

include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work1-tests ${SRC})

target_compile_features(work1-tests PRIVATE
    cxx_std_20
)

target_compile_options(work1-tests PRIVATE
    -Wall -O3
)

target_link_libraries(work1-tests PRIVATE
    work1
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
gtest_discover_tests(work1-tests)
