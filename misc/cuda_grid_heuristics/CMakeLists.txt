cmake_minimum_required(VERSION 3.24)
project(cuda-grid-heuristics LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_library(cuda-grid-heuristics INTERFACE)

target_include_directories(cuda-grid-heuristics INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(cuda-grid-heuristics INTERFACE
    cxx_std_20
)

target_link_libraries(cuda-grid-heuristics INTERFACE
    CUDA::cudart
)
