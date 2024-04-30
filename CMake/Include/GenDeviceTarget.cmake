
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

    set(SRC_ALL ${SRC_COMMON})
    if(${GEN_DEVICE_TARGET_MACRO} STREQUAL "MRAY_GPU_BACKEND_CUDA")
        set(SRC_ALL ${SRC_ALL}
            ${SRC_CUDA_MEMORY}
            ${SRC_CUDA}
            ${SRC_CUDA_ALGS})
    else()
        message(FATAL_ERROR "Unsupported Device Macro")
    endif()

    # IDE Filters
    source_group("CUDA/Memory" FILES ${SRC_CUDA_MEMORY})
    source_group("CUDA" FILES ${SRC_CUDA})
    source_group("CUDA/Algorithms" FILES ${SRC_CUDA_ALGS})
    source_group("" FILES ${SRC_COMMON})

    # Lib File
    set(TARGET_FULL_NAME "Device${GEN_DEVICE_TARGET_NAME}")
    add_library(${TARGET_FULL_NAME} STATIC ${SRC_ALL})

    # Make it public here, tracer will use this as well
    target_compile_definitions(${TARGET_FULL_NAME}
                               PUBLIC ${GEN_DEVICE_TARGET_MACRO})

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

    set(GEN_DEVICE_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)


endfunction()