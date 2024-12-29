cmake_minimum_required(VERSION 3.16)

project(pcg-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pcgSaad2003 pcgSaad2003.cpp)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

target_include_directories(pcgSaad2003 PRIVATE ${EIGEN3_INCLUDE_DIR})