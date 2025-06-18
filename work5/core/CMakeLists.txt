cmake_minimum_required(VERSION 3.24)
project(work5_core LANGUAGES CXX CUDA)

include(../../cmake/configs/cuda_arch.cmake)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(work5_core SHARED ${SRC})
add_library(work5::core ALIAS work5_core)

set_target_properties(work5_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_minimum_cuda_arch(work5_core 75)

target_compile_features(work5_core PRIVATE
    cxx_std_20
    cuda_std_20
)

target_compile_options(work5_core PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>
)

target_include_directories(work5_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(work5_core PRIVATE
    work2::core
    work4::core
    work1::cudagh
)
