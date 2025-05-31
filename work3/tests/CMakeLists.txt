cmake_minimum_required(VERSION 3.24)
project(work3_tests LANGUAGES CXX)

include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work3_tests ${SRC})

target_include_directories(work3_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work3_tests PRIVATE
    cxx_std_20
)

target_compile_options(work3_tests PRIVATE
    -Wall -O3
)

target_link_libraries(work3_tests PRIVATE
    work1::cudagh
    work2::core
    work2::tests_template
    work3::core
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
gtest_discover_tests(work3_tests)
