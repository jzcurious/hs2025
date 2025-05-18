cmake_minimum_required(VERSION 3.24)
project(work1-tests LANGUAGES CXX)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GoogleTest)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work1-tests ${SRC})

target_compile_features(work1-tests PRIVATE
    cxx_std_20
)

target_compile_options(work1-tests PRIVATE
    -Wall -O3
)

target_link_libraries(work1-tests PRIVATE
    work1
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
gtest_discover_tests(work1-tests)
