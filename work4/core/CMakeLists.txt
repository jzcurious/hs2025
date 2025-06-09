cmake_minimum_required(VERSION 3.24)
project(work4_core LANGUAGES CXX CUDA)

include(../../cmake/configs/cuda_arch.cmake)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(work4_core SHARED ${SRC})
add_library(work4::core ALIAS work4_core)

set_target_properties(work4_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_minimum_cuda_arch(work4_core 75)

target_compile_features(work4_core PRIVATE
    cxx_std_20
    cuda_std_20
)

target_compile_options(work4_core PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>
)

target_include_directories(work4_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(work4_core PRIVATE
    work2::core
    work3::core
    work1::cudagh
)
