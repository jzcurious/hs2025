cmake_minimum_required(VERSION 3.24)
project(work2-5-benchmarks LANGUAGES CXX)

include(FetchContent)

find_package(CUDAToolkit REQUIRED)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work2-5-benchmarks ${SRC})

target_compile_features(work2-5-benchmarks PRIVATE
    cxx_std_20
)

target_link_libraries(work2-5-benchmarks PRIVATE
    cuda-timer
    CUDA::cudart
    benchmark::benchmark
    Eigen3::Eigen
    work2
)
