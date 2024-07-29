
function(gen_device_target)

    # Parse Args
    set(options)
    set(oneValueArgs NAME MACRO)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_DEVICE_TARGET "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    # TODO: Add Host compilation target
    set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Device)

    # SOURCES
    set(SRC_CUDA_MEMORY
        ${CURRENT_SOURCE_DIR}/CUDA/DeviceMemoryCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/DeviceMemoryCUDA.cpp
        ${CURRENT_SOURCE_DIR}/CUDA/TextureCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/TextureCUDA.hpp
        ${CURRENT_SOURCE_DIR}/CUDA/TextureCUDA.cpp
        ${CURRENT_SOURCE_DIR}/CUDA/TextureViewCUDA.h)

    set(SRC_CUDA
        ${CURRENT_SOURCE_DIR}/CUDA/GPUSystemCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/GPUSystemCUDA.hpp
        ${CURRENT_SOURCE_DIR}/CUDA/GPUSystemCUDA.cpp
        ${CURRENT_SOURCE_DIR}/CUDA/DefinitionsCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/GPUAtomicCUDA.h)

    set(SRC_CUDA_ALGS
        ${CURRENT_SOURCE_DIR}/CUDA/AlgForwardCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/AlgReduceCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/AlgScanCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/AlgRadixSortCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/AlgBinaryPartitionCUDA.h
        ${CURRENT_SOURCE_DIR}/CUDA/AlgBinarySearchCUDA.h)

    set(SRC_ALGS
        ${CURRENT_SOURCE_DIR}/GPUAlgForward.h
        ${CURRENT_SOURCE_DIR}/GPUAlgGeneric.h
        ${CURRENT_SOURCE_DIR}/GPUAlgReduce.h
        ${CURRENT_SOURCE_DIR}/GPUAlgScan.h
        ${CURRENT_SOURCE_DIR}/GPUAlgRadixSort.h
        ${CURRENT_SOURCE_DIR}/GPUAlgBinaryPartition.h
        ${CURRENT_SOURCE_DIR}/GPUAlgBinarySearch.h)

    set(SRC_COMMON
        ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
        ${CURRENT_SOURCE_DIR}/GPUTexture.h
        ${CURRENT_SOURCE_DIR}/GPUTextureView.h
        ${CURRENT_SOURCE_DIR}/GPUMemory.h
        ${CURRENT_SOURCE_DIR}/GPUSystemForward.h
        ${CURRENT_SOURCE_DIR}/GPUSystem.h
        ${CURRENT_SOURCE_DIR}/GPUSystem.hpp
        ${CURRENT_SOURCE_DIR}/GPUDebug.h
        ${CURRENT_SOURCE_DIR}/GPUTypes.h
        ${CURRENT_SOURCE_DIR}/GPUAtomic.h)

    set(SRC_ALL
        ${SRC_COMMON}
        ${SRC_ALGS})

    if(${GEN_DEVICE_TARGET_MACRO} STREQUAL "MRAY_GPU_BACKEND_CUDA")
        set(SRC_ALL ${SRC_ALL}
            ${SRC_CUDA_MEMORY}
            ${SRC_CUDA}
            ${SRC_CUDA_ALGS})

        source_group("CUDA/Memory" FILES ${SRC_CUDA_MEMORY})
        source_group("CUDA" FILES ${SRC_CUDA})
        source_group("CUDA/Algorithms" FILES ${SRC_CUDA_ALGS})
    else()
        message(FATAL_ERROR "Unsupported Device Macro")
    endif()

    # Common IDE Filters
    source_group("Algorithms" FILES ${SRC_ALGS})
    source_group("" FILES ${SRC_COMMON})

    # Lib File
    set(TARGET_FULL_NAME "Device${GEN_DEVICE_TARGET_NAME}")
    add_library(${TARGET_FULL_NAME} STATIC ${SRC_ALL})

    # Make it public here, tracer will use this as well
    target_compile_definitions(${TARGET_FULL_NAME}
                               PUBLIC
                               ${GEN_DEVICE_TARGET_MACRO})

    target_link_libraries(${TARGET_FULL_NAME}
                          PUBLIC
                          CoreLib
                          TransientPool
                          CUDA::cuda_driver
                          PRIVATE
                          mray::meta_compile_opts
                          mray::cuda_extra_compile_opts)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION ON
                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})

    set(GEN_DEVICE_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)


endfunction()