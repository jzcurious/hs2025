cmake_minimum_required(VERSION 3.24)
project(hs2025 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(misc/cuda_timer)
add_subdirectory(misc/cuda_grid_heuristics)

add_subdirectory(work1)
add_subdirectory(works2_5)
