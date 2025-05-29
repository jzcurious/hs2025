cmake_minimum_required(VERSION 3.24)
project(work2_benchmarks LANGUAGES CXX)

include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work2_benchmarks ${SRC})

target_include_directories(work2_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work2_benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work2_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work2_benchmarks PRIVATE
    work2::core
    work1::cuda_timer
    CUDA::cudart
    benchmark::benchmark
    Eigen3::Eigen
)

add_library(work2_benchmarks_template INTERFACE)
add_library(work2::benchmarks_template ALIAS work2_benchmarks_template)

target_include_directories(work2_benchmarks_template INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work2_benchmarks_template INTERFACE
    cxx_std_20
)

target_link_libraries(work2_benchmarks_template INTERFACE
    work2::core
    work1::cuda_timer
    benchmark::benchmark
)
