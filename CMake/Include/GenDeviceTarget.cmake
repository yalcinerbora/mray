
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
    ################
    # CUDA RELATED #
    ################
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

    ###############
    # HIP RELATED #
    ###############
    set(SRC_HIP_MEMORY
        ${CURRENT_SOURCE_DIR}/HIP/DeviceMemoryHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/DeviceMemoryHIP.cpp
        ${CURRENT_SOURCE_DIR}/HIP/TextureHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/TextureHIP.hpp
        ${CURRENT_SOURCE_DIR}/HIP/TextureHIP.cpp
        ${CURRENT_SOURCE_DIR}/HIP/TextureViewHIP.h)

    set(SRC_HIP
        ${CURRENT_SOURCE_DIR}/HIP/GPUSystemHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/GPUSystemHIP.hpp
        ${CURRENT_SOURCE_DIR}/HIP/GPUSystemHIP.cpp
        ${CURRENT_SOURCE_DIR}/HIP/DefinitionsHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/GPUAtomicHIP.h)

    set(SRC_HIP_ALGS
        ${CURRENT_SOURCE_DIR}/HIP/AlgForwardHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/AlgReduceHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/AlgScanHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/AlgRadixSortHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/AlgBinaryPartitionHIP.h
        ${CURRENT_SOURCE_DIR}/HIP/AlgBinarySearchHIP.h)

    ###############
    # CPU RELATED #
    ###############
    set(SRC_CPU_MEMORY
        ${CURRENT_SOURCE_DIR}/CPU/DeviceMemoryCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/DeviceMemoryCPU.cpp
        ${CURRENT_SOURCE_DIR}/CPU/TextureCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/TextureCPU.hpp
        ${CURRENT_SOURCE_DIR}/CPU/TextureCPU.cpp
        ${CURRENT_SOURCE_DIR}/CPU/TextureViewCPU.h)

    set(SRC_CPU
        ${CURRENT_SOURCE_DIR}/CPU/GPUSystemCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/GPUSystemCPU.hpp
        ${CURRENT_SOURCE_DIR}/CPU/GPUSystemCPU.cpp
        ${CURRENT_SOURCE_DIR}/CPU/DefinitionsCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/GPUAtomicCPU.h)

    set(SRC_CPU_ALGS
        ${CURRENT_SOURCE_DIR}/CPU/AlgForwardCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/AlgReduceCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/AlgScanCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/AlgRadixSortCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/AlgBinaryPartitionCPU.h
        ${CURRENT_SOURCE_DIR}/CPU/AlgBinarySearchCPU.h)

    ##########
    # COMMON #
    ##########
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

    set(SRC_ALGS
        ${CURRENT_SOURCE_DIR}/GPUAlgForward.h
        ${CURRENT_SOURCE_DIR}/GPUAlgGeneric.h
        ${CURRENT_SOURCE_DIR}/GPUAlgReduce.h
        ${CURRENT_SOURCE_DIR}/GPUAlgScan.h
        ${CURRENT_SOURCE_DIR}/GPUAlgRadixSort.h
        ${CURRENT_SOURCE_DIR}/GPUAlgBinaryPartition.h
        ${CURRENT_SOURCE_DIR}/GPUAlgBinarySearch.h)

    set(SRC_ALL
        ${SRC_COMMON}
        ${SRC_ALGS})

    if(GEN_DEVICE_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_CUDA")
        set(SRC_ALL ${SRC_ALL}
            ${SRC_CUDA_MEMORY}
            ${SRC_CUDA}
            ${SRC_CUDA_ALGS})

        source_group("CUDA/Memory" FILES ${SRC_CUDA_MEMORY})
        source_group("CUDA" FILES ${SRC_CUDA})
        source_group("CUDA/Algorithms" FILES ${SRC_CUDA_ALGS})

    elseif(GEN_DEVICE_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_HIP")
        set(SRC_ALL ${SRC_ALL}
            ${SRC_HIP_MEMORY}
            ${SRC_HIP}
            ${SRC_HIP_ALGS})

        source_group("HIP/Memory" FILES ${SRC_HIP_MEMORY})
        source_group("HIP" FILES ${SRC_HIP})
        source_group("HIP/Algorithms" FILES ${SRC_HIP_ALGS})

    elseif(GEN_DEVICE_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_CPU")
        set(SRC_ALL ${SRC_ALL}
            ${SRC_CPU_MEMORY}
            ${SRC_CPU}
            ${SRC_CPU_ALGS})

        source_group("CPU/Memory" FILES ${SRC_CPU_MEMORY})
        source_group("CPU" FILES ${SRC_CPU})
        source_group("CPU/Algorithms" FILES ${SRC_CPU_ALGS})

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
                          PRIVATE
                          mray::meta_compile_opts)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON)

    if(GEN_DEVICE_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_CUDA")
        target_link_libraries(${TARGET_FULL_NAME}
                              PUBLIC
                              CUDA::cuda_driver
                              $<$<BOOL:${MRAY_ENABLE_TRACY}>:CUDA::cupti>
                              PRIVATE
                              mray::cuda_extra_compile_opts)

        # TODO: CUPIT dll is not near cudart.dll, so CUDA installation does not put it on path.
        # We copy it to our binary directory, but check if this is OK?
        if(MRAY_ENABLE_TRACY AND WIN32)
            # $<TARGET_FILE:CUDA::cupti> gives the lib file...
            # Configure time copying wont work folders may not be
            # created yet. CUDA 12.6
            # add_custom_command(TARGET ${TARGET_FULL_NAME} POST_BUILD
            #                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            #                    "$<TARGET_FILE:CUDA::cupti>"
            #                    ${MRAY_CONFIG_BIN_DIRECTORY}
            #                    COMMENT "Copying CUPTI dll")
        endif()

        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    elseif(GEN_DEVICE_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_CPU")
        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              INTERPROCEDURAL_OPTIMIZATION_RELEASE
                              ${MRAY_HOST_BACKEND_IPO_MODE})
    endif()
    # elseif(${GEN_DEVICE_TARGET_MACRO} STREQUAL "MRAY_GPU_BACKEND_HIP")
    #     # target_link_libraries(${TARGET_FULL_NAME}
    #     #                       PUBLIC
    #     #                       hip::host)
    # else()
    #     message(FATAL_ERROR "Unsupported Device Macro")
    # endif()

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})

    set(GEN_DEVICE_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)


endfunction()