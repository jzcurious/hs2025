cmake_minimum_required(VERSION 3.24)
project(work2-5-tests LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

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

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work2-5-tests ${SRC})

target_compile_features(work2-5-tests PRIVATE
    cxx_std_20
    cuda_std_20
)

target_compile_options(work2 PRIVATE
    -Wall -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ccbin clang++>
)

target_link_libraries(work2-5-tests PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    work2
)

enable_testing()
gtest_discover_tests(work2-5-tests)
