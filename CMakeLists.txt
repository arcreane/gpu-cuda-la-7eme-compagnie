cmake_minimum_required(VERSION 3.23)
project(QTProjectAppli_7ecompagnie LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(hello_build src/hello_build.cpp)

find_package(CUDAToolkit REQUIRED)
add_executable(hello_cuda src/hello_cuda.cu)
target_link_libraries(hello_cuda PRIVATE CUDA::cudart)

set_target_properties(hello_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(check_headers src/check_headers.cpp)
target_include_directories(check_headers PRIVATE lib)

# --- lib CPU ---
add_library(compute_cpu lib/compute_cpu.cpp)
target_include_directories(compute_cpu PUBLIC lib)
add_library(compute_cuda lib/compute_cuda.cu)
target_include_directories(compute_cuda PUBLIC lib)
target_link_libraries(compute_cuda PRIVATE CUDA::cudart)

# --- test console CPU ---
add_executable(test_cpu_only src/test_cpu_only.cpp)
target_include_directories(test_cpu_only PRIVATE lib)
target_link_libraries(test_cpu_only PRIVATE compute_cpu)

