cmake_minimum_required(VERSION 3.24)
project(work2-5-tests LANGUAGES CUDA)

include(../../cmake/deps/gtest.cmake)
include(../../cmake/deps/eigen.cmake)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work2-5-tests ${SRC})

target_compile_features(work2-5-tests PRIVATE
    cuda_std_20
)

target_compile_options(work2-5-tests PRIVATE
    -Wall -O3
    -Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr
)

target_link_libraries(work2-5-tests PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    work2
)

enable_testing()
gtest_discover_tests(work2-5-tests)
