cmake_minimum_required(VERSION 3.25)
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
    CUDA_ARCHITECTURES 75
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
set(RAYLIB_LIB "C:/raylib/raylib/build_msvc/raylib/Release/raylib.lib")
set(RAYLIB_INCLUDE_DIR "C:/raylib/raylib/src")

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

