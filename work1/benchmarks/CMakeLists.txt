cmake_minimum_required(VERSION 3.24)
project(work1-benchmarks LANGUAGES CXX)

include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work1-benchmarks ${SRC})

target_compile_features(work1-benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work1-benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work1-benchmarks PRIVATE
    work1
    cuda-timer
    CUDA::cudart
    benchmark::benchmark
    Eigen3::Eigen
    cuda-grid-heuristics
)
