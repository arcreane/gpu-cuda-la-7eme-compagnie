cmake_minimum_required(VERSION 3.25)
project(GPU_Particles LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# --------------------------------
# Bibliothèque CPU
# --------------------------------
add_library(compute_cpu
    lib/compute_cpu.cpp
)

target_include_directories(compute_cpu PUBLIC lib)

# --------------------------------
# Bibliothèque CUDA (backend GPU)
# --------------------------------
add_library(compute_cuda
    lib/compute_cuda.cu
)

target_include_directories(compute_cuda PUBLIC lib)

set_target_properties(compute_cuda PROPERTIES
    CUDA_ARCHITECTURES 75
)

target_link_libraries(compute_cuda PUBLIC CUDA::cudart)

# --------------------------------
# Fabrique de backends
# (pour l'instant CPU only)
# --------------------------------
add_library(backend_factory
    lib/backend_factory.cpp
    lib/compute.hpp
    lib/particles_types.hpp
)

target_link_libraries(backend_factory PUBLIC compute_cpu)
target_include_directories(backend_factory PUBLIC lib)

# --------------------------------
# SimWorld
# --------------------------------
add_library(sim_world
    lib/sim_world.cpp
    lib/sim_world.hpp
)

target_link_libraries(sim_world PUBLIC backend_factory)
target_include_directories(sim_world PUBLIC lib)

# --------------------------------
# EXE de tests CPU-only
# --------------------------------
add_executable(check_headers src/check_headers.cpp)
target_link_libraries(check_headers PRIVATE backend_factory)

add_executable(test_cpu_only src/test_cpu_only.cpp)
target_link_libraries(test_cpu_only PRIVATE backend_factory)

add_executable(test_perf src/test_perf.cpp)
target_link_libraries(test_perf PRIVATE backend_factory)

add_executable(test_world src/test_world.cpp)
target_link_libraries(test_world PRIVATE sim_world)

# --------------------------------
# RAYLIB DEMO (CPU pour l'instant)
# --------------------------------
set(RAYLIB_LIB "C:/raylib/raylib/build_msvc/raylib/Release/raylib.lib")
set(RAYLIB_INCLUDE_DIR "C:/raylib/raylib/src")

add_executable(raylib_demo
    src/main_raylib.cpp
)

target_include_directories(raylib_demo PRIVATE ${RAYLIB_INCLUDE_DIR})

target_link_libraries(raylib_demo PRIVATE
    sim_world
    ${RAYLIB_LIB}
    winmm
)

