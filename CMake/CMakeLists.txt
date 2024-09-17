cmake_minimum_required(VERSION 3.26)

# Deduplicate archivies when linking
cmake_policy(VERSION 3.26)

# Some things that will be fed to both CMAKE and C++
set(MRAY_PROJECT_NAME "MRay")
set(MRAY_PROJECT_DESCRIPTION "GPU-based rendering framework / renderer.")
set(MRAY_VERSION_MAJOR "0")
set(MRAY_VERSION_MINOR "1")
set(MRAY_VERSION_PATCH "0")

# All CMake Related files is in the current folder
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Include")

# Do not use in-source builds
include("PreventInSourceBuilds")

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;SanitizeR;Release CACHE STRING "Build configurations. SanitizeR has sanitized output")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single config generators")

# Architecture
# Use a drop-down menu for selecting Compute Capability
set(CMAKE_CUDA_ARCHITECTURES "native"
    CACHE STRING "Architecture chosen by the user at CMake configure time")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES
             PROPERTY STRINGS native 52 60 61 70 72 75 86 89 90 all all-major)

set(MRAY_DEVICE_BACKEND "MRAY_GPU_BACKEND_CUDA"
    CACHE STRING "Device backend chosen by the user at CMake configure time")
set_property(CACHE MRAY_DEVICE_BACKEND
             PROPERTY STRINGS MRAY_GPU_BACKEND_CUDA)

option(MRAY_ENABLE_HW_ACCELERATION "Try to find, compile and and use HW accelration (i.e. this means OptiX for CUDA)" OFF)
option(MRAY_BUILD_TESTS "Build Tests" ON)
option(MRAY_EXPORT_COMPILE_COMMANDS "Export compile commands for IWYU" ON)
option(MRAY_ENABLE_PCH "Enable precompiled headers functionality" OFF)

# Globals Common for All Projects
# TODO: Align CUDA Standard with CXX Standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Set compile command export
if(MRAY_EXPORT_COMPILE_COMMANDS)
    set(ENV{CMAKE_EXPORT_COMPILE_COMMANDS} ON)
else()
    set(ENV{CMAKE_EXPORT_COMPILE_COMMANDS} OFF)
endif()

# Filter ALL_BUILD and ZERO_CHECK projects
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(MRAY_PREDEFINED_TARGETS_FOLDER "CustomTargets")
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER ${MRAY_PREDEFINED_TARGETS_FOLDER})

# ====================== #
#     Project Start      #
# ====================== #
project(${MRAY_PROJECT_NAME}
        LANGUAGES CUDA CXX
        VERSION "${MRAY_VERSION_MAJOR}.${MRAY_VERSION_MINOR}.${MRAY_VERSION_PATCH}"
        DESCRIPTION ${MRAY_PROJECT_DESCRIPTION}
)

include("Macros")
include("AddPCH")
# CMAKE Project Definitions and Source, Library Folders are separate
# Directories
set(MRAY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
cmake_path(NORMAL_PATH MRAY_DIRECTORY)
# Populate the meta directories for this project hierarchy
generate_platform_name()
populate_mray_directory_variables(${MRAY_DIRECTORY})

# Use currently selected compiler for CUDA
# TODO: Check why this is not set?
if(MSVC)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    # This should be set automatically I think...
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)
endif()

# After project is set and nvcc is found invoke find CUDA toolkit
# we will use curand for sampling
find_package(CUDAToolkit REQUIRED)
# Bug on cmake maybe this variable is not advanced
mark_as_advanced(CUDAToolkit_CUPTI_INCLUDE_DIR)

# Force Colored Output on Ninja
set(CMAKE_COLOR_DIAGNOSTICS ON)

# Compile Option Targets
# Generates
#   mray::meta_compile_opts
#   mray::cuda_extra_compile_opts
include("CompilerOptions")

# Generate Configuration Header
cmake_path(GET CMAKE_CXX_COMPILER STEM MRAY_COMPILER_NAME)
cmake_path(GET CMAKE_CUDA_COMPILER STEM MRAY_CUDA_COMPILER_NAME)

# Determine gpu platform via MRAY_DEVICE_BACKEND
if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")
    set(MRAY_GPU_PLATFORM_NAME "CUDA")
else()
    message(FATAL_ERROR "Unknown Device backend!"
                        "Please set MRAY_DEVICE_BACKEND to a valid backend")
endif()

configure_file(${MRAY_SOURCE_DIRECTORY}/mray_cmake.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/mray_gen/mray_cmake.h
               NO_SOURCE_PERMISSIONS
               @ONLY)
target_include_directories(meta_compile_opts
                           SYSTEM INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/mray_gen")

# After Everything is set
# Include NVCC compile to ptx routine
# Since Optix programs require precompiled *.ptx files
if(MRAY_USE_OPTIX)
    include("NVCCCompilePTX")
endif()

# Enable LTO / Device-LTO
# Device LTO is slow so we disabled it.
# TODO: Enable LTO via config maybe?
# or enable LTO only for Host side (check if this is possible)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE FASLE)
#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

# PCH
add_subdirectory(PrecompiledHeaders)

# External Libs
add_subdirectory(Lib)

# BUILDS
# Libs
add_subdirectory(Core)

# DLLs
add_subdirectory(TransientPool)
add_subdirectory(SceneLoaderMRay)
add_subdirectory(MeshLoader)
add_subdirectory(ImageLoader)
add_subdirectory(GFGConverter)
add_subdirectory(Visor)

# Generate Targets
add_subdirectory(TracerDevice)
#add_subdirectory(TracerHost)

# Execs
add_subdirectory(MRay)

# Tests
if(MRAY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
