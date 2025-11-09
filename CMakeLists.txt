cmake_minimum_required(VERSION 3.23)
project(QTProjectAppli_7ecompagnie LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(hello_build src/hello_build.cpp)

find_package(CUDAToolkit REQUIRED)
add_executable(hello_cuda src/hello_cuda.cu)
target_link_libraries(hello_cuda PRIVATE CUDA::cudart)
# (optionnel) évite la séparabilité pour simplifier
set_target_properties(hello_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(check_headers src/check_headers.cpp)
target_include_directories(check_headers PRIVATE lib)
