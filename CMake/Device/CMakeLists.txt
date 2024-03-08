set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Device)

# SOURCES
set(SRC_CUDA_MEMORY
    ${CURRENT_SOURCE_DIR}/DeviceMemoryCUDA.h
    ${CURRENT_SOURCE_DIR}/DeviceMemoryCUDA.cpp
    ${CURRENT_SOURCE_DIR}/TextureCUDA.h
    ${CURRENT_SOURCE_DIR}/TextureCUDA.hpp
    ${CURRENT_SOURCE_DIR}/TextureCUDA.cpp
    ${CURRENT_SOURCE_DIR}/TextureViewCUDA.h)

set(SRC_CUDA
    ${CURRENT_SOURCE_DIR}/GPUSystemCUDA.h
    ${CURRENT_SOURCE_DIR}/GPUSystemCUDA.hpp
    ${CURRENT_SOURCE_DIR}/GPUSystemCUDA.cu
    ${CURRENT_SOURCE_DIR}/DefinitionsCUDA.h)

set(SRC_CUDA_ALGS
    ${CURRENT_SOURCE_DIR}/AlgReduceCUDA.h
    ${CURRENT_SOURCE_DIR}/AlgScanCUDA.h
    ${CURRENT_SOURCE_DIR}/AlgRadixSortCUDA.h
    ${CURRENT_SOURCE_DIR}/AlgBinaryPartitionCUDA.h
    ${CURRENT_SOURCE_DIR}/AlgBinarySearchCUDA.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/GPUAlgorithms.h
    ${CURRENT_SOURCE_DIR}/GPUSystemForward.h
    ${CURRENT_SOURCE_DIR}/GPUSystem.h
    ${CURRENT_SOURCE_DIR}/GPUSystem.hpp
    ${CURRENT_SOURCE_DIR}/GPUTypes.h)

set(SRC_ALL
    ${SRC_CUDA_MEMORY}
    ${SRC_CUDA}
    ${SRC_CUDA_ALGS}
    ${SRC_COMMON})

# IDE Filters
source_group("CUDA/Memory" FILES ${SRC_CUDA_MEMORY})
source_group("CUDA" FILES ${SRC_CUDA})
source_group("CUDA/Algorithms" FILES ${SRC_CUDA_ALGS})
source_group("" FILES ${SRC_COMMON})

# DLL File
add_library(DeviceLib STATIC ${SRC_ALL})

target_link_libraries(DeviceLib
                      PUBLIC
                      CoreLib
                      TransientPool
                      CUDA::cuda_driver
                      PRIVATE
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts)

set_target_properties(DeviceLib PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Copy DLLS
# if(MSVC)
#     add_custom_command(TARGET CoreLib POST_BUILD
#                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                        "${MRAY_CONFIG_LIB_DIRECTORY}/spdlog$<$<CONFIG:Debug>:d>.dll"
#                        "${MRAY_CONFIG_LIB_DIRECTORY}/ZLIB$<$<CONFIG:Debug>:d>1.dll"
#                        ${MRAY_CONFIG_BIN_DIRECTORY})
# endif()