cmake_minimum_required(VERSION 3.20)

project(ort_link_probe LANGUAGES CXX)

find_package(onnxruntime CONFIG REQUIRED)

add_executable(ort_link_probe main.cpp)
target_compile_features(ort_link_probe PRIVATE cxx_std_20)
target_link_libraries(ort_link_probe PRIVATE onnxruntime::onnxruntime)
