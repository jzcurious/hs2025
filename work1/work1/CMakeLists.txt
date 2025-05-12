cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(work1 SHARED ${SRC})

target_compile_definitions(work1 PRIVATE
    "DEBUG=$<CONFIG:Debug>"
)

target_compile_features(work1 PRIVATE
    cuda_std_20
)

set_target_properties(work1 PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(work1 PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

target_include_directories(work1 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)
