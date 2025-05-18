cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" ON)

add_subdirectory(work1)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif()
