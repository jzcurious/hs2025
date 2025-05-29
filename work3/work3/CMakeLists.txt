cmake_minimum_required(VERSION 3.24)
project(work4 LANGUAGES CXX CUDA)

include(../../cmake/configs/cuda_arch.cmake)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(work4 SHARED ${SRC})

set_target_properties(work4 PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_minimum_cuda_arch(work4 75)

target_compile_features(work4 PRIVATE
    cxx_std_20
    cuda_std_20
)

target_compile_options(work4 PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>
)

target_include_directories(work4 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(work4 PRIVATE
    work2
    cuda-grid-heuristics
)
