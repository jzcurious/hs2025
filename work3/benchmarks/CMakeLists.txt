cmake_minimum_required(VERSION 3.24)
project(work3_benchmarks LANGUAGES CXX)

include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work3_benchmarks ${SRC})

target_include_directories(work3_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work3_benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work3_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work3_benchmarks PRIVATE
    work2::core
    work2::benchmarks_template
    work3::core
    CUDA::cudart
    benchmark::benchmark
    Eigen3::Eigen
)
