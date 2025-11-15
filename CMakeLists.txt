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

# --- lib CUDA ---
add_library(compute_cuda lib/compute_cuda.cu)
target_include_directories(compute_cuda PUBLIC lib)
target_link_libraries(compute_cuda PRIVATE CUDA::cudart)

# --- fabrique ---
add_library(backend_factory lib/backend_factory.cpp)
target_include_directories(backend_factory PUBLIC lib)
target_link_libraries(backend_factory PUBLIC compute_cpu)
# (plus tard on ajoutera compute_cuda ici quand la fabrique utilisera vraiment le GPU)

add_library(sim_world lib/sim_world.cpp)
target_include_directories(sim_world PUBLIC lib)
target_link_libraries(sim_world PUBLIC backend_factory)

# --- test console ---
add_executable(test_cpu_only src/test_cpu_only.cpp)
target_include_directories(test_cpu_only PRIVATE lib)
target_link_libraries(test_cpu_only PRIVATE backend_factory)

add_executable(test_perf src/test_perf.cpp)
target_include_directories(test_perf PRIVATE lib)
target_link_libraries(test_perf PRIVATE backend_factory)

add_executable(test_world src/test_world.cpp)
target_include_directories(test_world PRIVATE lib)
target_link_libraries(test_world PRIVATE sim_world)
