cmake_minimum_required(VERSION 3.25)

# Allow unknown compiler versions
set(CMAKE_CXX_COMPILER_ID_PLATFORM_CONTENT "")
if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()

# Force CUDA 12.9 BEFORE project()
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

project(GPU_Particles LANGUAGES CXX CUDA)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------
# Qt & CUDA setup
# ---------------------------------------------------------------------
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_PREFIX_PATH "C:/Qt/6.9.3/msvc2022_64")

find_package(CUDAToolkit REQUIRED)
find_package(Qt6 COMPONENTS Widgets REQUIRED)

# ---------------------------------------------------------------------
# CPU Backend
# ---------------------------------------------------------------------
add_library(compute_cpu lib/compute_cpu.cpp)
target_include_directories(compute_cpu PUBLIC lib)

# ---------------------------------------------------------------------
# GPU Backend
# ---------------------------------------------------------------------
add_library(compute_cuda lib/compute_cuda.cu)
target_include_directories(compute_cuda PUBLIC lib)

set_target_properties(compute_cuda PROPERTIES
    CUDA_ARCHITECTURES "90-real;90-virtual"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(compute_cuda PUBLIC CUDA::cudart)

# ---------------------------------------------------------------------
# Factory/Bakcend
# ---------------------------------------------------------------------
add_library(backend_factory
    lib/backend_factory.cpp
    lib/compute.hpp
    lib/particles_types.hpp
)

target_include_directories(backend_factory PUBLIC lib)
target_link_libraries(backend_factory PUBLIC compute_cpu compute_cuda)

# ---------------------------------------------------------------------
# SimWorld
# ---------------------------------------------------------------------
add_library(sim_world
    lib/sim_world.cpp
    lib/sim_world.hpp
)

target_include_directories(sim_world PUBLIC lib)
target_link_libraries(sim_world PUBLIC backend_factory)

# ---------------------------------------------------------------------
# Raylib (used in Qt)
# ---------------------------------------------------------------------
set(RAYLIB_LIB "C:/Users/henri/Downloads/raylib-5.5_win64_msvc16/raylib-5.5_win64_msvc16/lib/raylib.lib")
set(RAYLIB_INCLUDE_DIR "C:/Users/henri/Downloads/raylib-5.5_win64_msvc16/raylib-5.5_win64_msvc16/include")

# ---------------------------------------------------------------------
# EXECUTABLE : qt_interface
# ---------------------------------------------------------------------
add_executable(qt_interface
    qt_interface/main.cpp
    qt_interface/projet_interface.cpp
    qt_interface/projet_interface.h
    qt_interface/projet_interface.ui
    qt_interface/projet_interface.qrc
    qt_interface/ParticleView.cpp
    qt_interface/ParticleView.h
    qt_interface/app_icon.rc
)

target_include_directories(qt_interface PRIVATE ${RAYLIB_INCLUDE_DIR})

target_link_libraries(qt_interface PRIVATE
    Qt6::Widgets
    sim_world
    ${RAYLIB_LIB}
    winmm
)

# ---------------------------------------------------------------------
# EXECUTABLE : benchmark_demo
# ---------------------------------------------------------------------
add_executable(benchmark_demo
    benchmark_demo.cpp
)

target_include_directories(benchmark_demo PRIVATE ${RAYLIB_INCLUDE_DIR})

target_link_libraries(benchmark_demo PRIVATE
    sim_world
    ${RAYLIB_LIB}
    winmm
)


