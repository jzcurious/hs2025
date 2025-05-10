cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(work1)
add_subdirectory(tests)
add_subdirectory(benchmarks)
