cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(misc/cuda_timer)
add_subdirectory(work1)
