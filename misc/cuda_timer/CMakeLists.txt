cmake_minimum_required(VERSION 3.24)
project(cuda-timer LANGUAGES CXX)

include(FetchContent)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(cuda-timer STATIC ${SRC})

target_include_directories(cuda-timer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_compile_features(cuda-timer PRIVATE
    cxx_std_20
)

target_link_libraries(cuda-timer PRIVATE
    CUDA::cudart
)
