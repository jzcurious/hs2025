cmake_minimum_required(VERSION 3.24)
project(work2 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(work2 SHARED ${SRC})

target_compile_definitions(work2 PRIVATE
    "DEBUG=$<CONFIG:Debug>"
)

target_compile_features(work2 PRIVATE
    cxx_std_20
    cuda_std_20
)

set_target_properties(work2 PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(work2 PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

target_include_directories(work2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)
