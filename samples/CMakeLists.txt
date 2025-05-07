cmake_minimum_required(VERSION 3.24)
project(proj-samples LANGUAGES CXX)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
add_executable(proj-samples ${SRC})

target_include_directories(proj-samples PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)
target_link_libraries(proj-samples PRIVATE
    proj
)
