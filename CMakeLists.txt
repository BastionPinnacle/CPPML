cmake_minimum_required(VERSION 3.15)

project(cppml LANGUAGES CXX)

add_library(compiler_flags INTERFACE)

target_compile_features(compiler_flags INTERFACE cxx_std_11)

add_subdirectory(src)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE cppml)
