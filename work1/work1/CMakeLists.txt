cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX CUDA)

include(FetchContent)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(work1 SHARED ${SRC})

target_compile_features(work1 PRIVATE
    cxx_std_20
    cuda_std_20
)

set_target_properties(work1 PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions(work1 PRIVATE
    "DEBUG=$<CONFIG:Debug>"
)

target_include_directories(work1 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_options(work1 PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)
