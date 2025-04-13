cmake_minimum_required(VERSION 3.25)
project(TimeSeriesX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(backend)
add_subdirectory(bindings)