cmake_minimum_required(VERSION 3.24)
project(work2_tests LANGUAGES CXX)

include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work2_tests ${SRC})

target_include_directories(work2_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work2_tests PRIVATE
    cxx_std_20
)

target_compile_options(work2_tests PRIVATE
    -Wall -O3
)

target_link_libraries(work2_tests PRIVATE
    work2::core
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
    work1::cudagh
)

enable_testing()
gtest_discover_tests(work2_tests)

add_library(work2_tests_template INTERFACE)
add_library(work2::tests_template ALIAS work2_tests_template)

target_include_directories(work2_tests_template INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work2_tests_template INTERFACE
    cxx_std_20
)

target_link_libraries(work2_tests_template INTERFACE
    work2::core
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
    work1::cudagh
)
