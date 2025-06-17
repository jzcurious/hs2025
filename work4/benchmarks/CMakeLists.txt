cmake_minimum_required(VERSION 3.24)
project(work4_benchmarks LANGUAGES CXX)

include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work4_benchmarks ${SRC})

target_include_directories(work4_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(work4_benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work4_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work4_benchmarks PRIVATE
    work2::core
    work3::core
    work4::core
    work2::benchmarks_template
    CUDA::cudart
    benchmark::benchmark
)
