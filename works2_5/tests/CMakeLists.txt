cmake_minimum_required(VERSION 3.24)
project(work2-5-tests LANGUAGES CXX)

include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work2-5-tests ${SRC})

target_compile_features(work2-5-tests PRIVATE
    cxx_std_20
)

target_compile_options(work2-5-tests PRIVATE
    -Wall -O3
)

target_link_libraries(work2-5-tests PRIVATE
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
    work2
    work3
    work4
)

enable_testing()
gtest_discover_tests(work2-5-tests)
