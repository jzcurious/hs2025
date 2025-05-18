cmake_minimum_required(VERSION 3.24)
project(works2-5-benchmarks LANGUAGES CXX)

include(../../cmake/deps/gbench.cmake)
include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(works2-5-benchmarks ${SRC})

target_compile_features(works2-5-benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(works2-5-benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(works2-5-benchmarks PRIVATE
    cuda-timer
    CUDA::cudart
    benchmark::benchmark
    Eigen3::Eigen
    work2
)
