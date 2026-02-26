cmake_minimum_required(VERSION 3.18)
project(vamospp LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core cpp/bindings/vamospp_module.cpp)
target_compile_features(_core PRIVATE cxx_std_17)
target_compile_definitions(_core PRIVATE NB_STATIC)

install(TARGETS _core LIBRARY DESTINATION vamospp)
