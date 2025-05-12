cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(W1 "Build work1" OFF)
option(W2 "Build work1" OFF)

if(W1)
    add_subdirectory(misc/cuda_timer)
    add_subdirectory(work1)
else()
    add_subdirectory(misc/cuda_timer)
    add_subdirectory(work1)
endif()
