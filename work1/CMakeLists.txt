cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_subdirectory(work1)
add_subdirectory(tests)
add_subdirectory(benchmarks)
