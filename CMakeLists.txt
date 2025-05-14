cmake_minimum_required(VERSION 3.24)
project(hs2025 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# option(W1 "Build work1" OFF)
# option(W2_5 "Build works 2-5" OFF)

# if(W1)
#     add_subdirectory(misc/cuda_timer)
#     add_subdirectory(work1)
# else()
#     add_subdirectory(misc/cuda_timer)
#     add_subdirectory(work1)
# endif()

add_subdirectory(misc/cuda_timer)
add_subdirectory(work1)
add_subdirectory(works2_5)
