cmake_minimum_required(VERSION 3.26...4.0)

# Deduplicate archives when linking
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
set(CMAKE_CONFIGURATION_TYPES Debug;SanitizeR;Release CACHE STRING "Build configurations. SanitizeR is a config with address sanitizer/release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single config generators")

# Control over your own destiny!
set(CMAKE_C_FLAGS_RELEASE    "" CACHE STRING "")
set(CMAKE_C_FLAGS_DEBUG      "" CACHE STRING "")
set(CMAKE_C_FLAGS            "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE  "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG    "" CACHE STRING "")
set(CMAKE_CXX_FLAGS          "" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG   "" CACHE STRING "")
set(CMAKE_CUDA_FLAGS         "" CACHE STRING "")

# Add build runpath $ORIGIN
# Emulate windows style
set(CMAKE_BUILD_RPATH $ORIGIN)

# Architecture
# Use a drop-down menu for selecting Compute Capability
set(MRAY_DEVICE_BACKEND "MRAY_GPU_BACKEND_CUDA"
    CACHE STRING "Device backend chosen by the user at CMake configure time")
set_property(CACHE MRAY_DEVICE_BACKEND
             PROPERTY STRINGS
             MRAY_GPU_BACKEND_CUDA
             MRAY_GPU_BACKEND_HIP)

set(MRAY_HOST_ARCH "MRAY_HOST_ARCH_BASIC"
    CACHE STRING "Host architecture chosen by the user at CMake configure time")
set_property(CACHE MRAY_HOST_ARCH
             PROPERTY STRINGS
             MRAY_HOST_ARCH_BASIC
             MRAY_HOST_ARCH_AVX2
             MRAY_HOST_ARCH_AVX512)

# Globals Common for All Projects
# TODO: Align CUDA Standard with CXX Standard
if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")

    set(CMAKE_CUDA_ARCHITECTURES "native"
        CACHE STRING "Architecture chosen by the user at CMake configure time")
    set_property(CACHE CMAKE_CUDA_ARCHITECTURES
                 PROPERTY STRINGS native 52 60 61 70 72 75 86 89 90 all all-major)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    set(MRAY_PROJECT_GPU_LANGUAGES CUDA)
    set(MRAY_GPU_PLATFORM_NAME "CUDA")

elseif(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_HIP")

    # TODO: Remove this HIP will only target AMD gpus
    set(CMAKE_HIP_PLATFORM "amd")
    mark_as_advanced(CMAKE_HIP_PLATFORM)

    set(CMAKE_HIP_ARCHITECTURES "gfx1030;gfx1100;gfx1101;gfx1102")

    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED True)
    set(CMAKE_HIP_RUNTIME_LIBRARY Shared)
    set(MRAY_PROJECT_GPU_LANGUAGES HIP)
    set(MRAY_GPU_PLATFORM_NAME "HIP")
    # Add .cu files as source files for HIP
    set(CMAKE_USER_MAKE_RULES_OVERRIDE_HIP "HIPFileExtExtras")
else()
    message(FATAL_ERROR "Unknown Device backend!"
                        "Please set MRAY_DEVICE_BACKEND to a valid backend")
endif()

# Secret Host backend and Device disabling
# These are secret because CPU backend is not implemented
# and disable option is to develop CPU-backend faster (to not waste compilation time)
option(MRAY_BUILD_HOST_BACKEND "Compile CPU backend in addition to GPU backend" OFF)
# Secrect Device disable option
option(MRAY_DISABLE_DEVICE_BACKEND "Disable GPU backend" OFF)
mark_as_advanced(MRAY_BUILD_HOST_BACKEND MRAY_DISABLE_DEVICE_BACKEND)

option(MRAY_ENABLE_HW_ACCELERATION "Try to find, compile and and use HW acceleration (i.e. this means OptiX for CUDA)" OFF)
option(MRAY_BUILD_TESTS "Build Tests" ON)
option(MRAY_BUILD_VISOR "Build Visor (Interactive Vulkan Viewer)" ON)
option(MRAY_EXPORT_COMPILE_COMMANDS "Export compile commands for IWYU" ON)
option(MRAY_ENABLE_PCH "Enable precompiled headers functionality" OFF)
option(MRAY_BUILD_DOCS "Build Documentation" OFF)
option(MRAY_ENABLE_USD "Enable usd scene file reading functionality" OFF)

# Sanitizer Modes for debugging etc.
set(MRAY_SANITIZER_MODE "address"
    CACHE STRING
    "Mode for the SanitizeR target, possible values \
    \"address\", \"undefined\", \"memory\", and     \
    \"thread\"")
set_property(CACHE MRAY_SANITIZER_MODE
             PROPERTY STRINGS
             "address" "undefined"
             "memory" "thread")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
        LANGUAGES ${MRAY_PROJECT_GPU_LANGUAGES} CXX
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
if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")
    find_package(CUDAToolkit REQUIRED)
    # Bug on cmake maybe this variable is not in "advanced"
    mark_as_advanced(CUDAToolkit_CUPTI_INCLUDE_DIR)

# elseif(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_HIP")
#     find_package(hip REQUIRED)
endif()

# Force Colored Output on Ninja
set(CMAKE_COLOR_DIAGNOSTICS ON)

# Compile Option Targets
# Generates
#   mray::meta_compile_opts
#   mray::cuda_extra_compile_opts
include("CompilerOptions")

# Generate Configuration Header
cmake_path(GET CMAKE_CXX_COMPILER STEM MRAY_COMPILER_NAME)
if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")
    cmake_path(GET CMAKE_CUDA_COMPILER STEM MRAY_GPU_COMPILER_NAME)
    set(MRAY_GPU_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
elseif(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_HIP")
    cmake_path(GET CMAKE_HIP_COMPILER STEM MRAY_GPU_COMPILER_NAME)
    set(MRAY_GPU_COMPILER_VERSION ${CMAKE_HIP_COMPILER_VERSION})
endif()

configure_file(${MRAY_SOURCE_DIRECTORY}/mray_cmake.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/mray_gen/mray_cmake.h
               NO_SOURCE_PERMISSIONS
               @ONLY)
target_include_directories(meta_compile_opts
                           SYSTEM INTERFACE
                           "${CMAKE_CURRENT_BINARY_DIR}/mray_gen")

# Enable LTO / Device-LTO
# Device LTO is slow so we disabled it.
# TODO: Enable LTO via config maybe?
# or enable LTO only for Host side (check if this is possible)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE FASLE)
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

if(MRAY_ENABLE_USD)
    add_subdirectory(SceneLoaderUSD)
endif()

add_subdirectory(SceneLoaderMRay)
add_subdirectory(MeshLoader)
add_subdirectory(ImageLoader)
add_subdirectory(GFGConverter)

if(MRAY_BUILD_VISOR)
    add_subdirectory(Visor)
endif()

# Generate Tracers
add_subdirectory(TracerKernelGen)
# Skip device targets if disabled
# It is for debugging reducing compile times
# Thus, the parameter is inverted
if(NOT MRAY_DISABLE_DEVICE_BACKEND)
    add_subdirectory(TracerCUDA)
    #add_subdirectory(TracerHIP)
endif()

# Enable Host target if requested
if(MRAY_BUILD_HOST_BACKEND)
    add_subdirectory(TracerCPU)
endif()

# Execs
add_subdirectory(MRay)

# Tests
if(MRAY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()

# Docs
if(MRAY_BUILD_DOCS)
    add_subdirectory(Docs)
endif()
