cmake_minimum_required(VERSION 3.24)
project(work4_tests LANGUAGES CXX)

include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work4_tests ${SRC})

target_include_directories(work4_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work4_tests PRIVATE
    cxx_std_20
)

target_compile_options(work4_tests PRIVATE
    -Wall -O3
)

target_link_libraries(work4_tests PRIVATE
    work1::cudagh
    work2::core
    work3::core
    work4::core
    work2::tests_template
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
gtest_discover_tests(work4_tests)
