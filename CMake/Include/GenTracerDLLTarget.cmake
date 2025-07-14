include("GenDeviceTarget")
include("GenTracerTarget")
include("GenTracerTypesAndKernels")
include("ChangeDeviceSourceFileLanguage")

function(gen_tracer_dll_target)

    set(options ENABLE_HW_ACCEL)
    set(oneValueArgs NAME MACRO)
    set(multiValueArgs)
    cmake_parse_arguments(GEN_TRACER_DLL "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/TracerDLL)

    set(SRC_RENDERERS_PATH_TRACER
        ${CURRENT_SOURCE_DIR}/PathTracerRenderer.cu
        ${CURRENT_SOURCE_DIR}/PathTracerRenderer.h
        ${CURRENT_SOURCE_DIR}/PathTracerRendererShaders.h)

    set(SRC_COMMON
        ${CURRENT_SOURCE_DIR}/EntryPoint.h
        ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
        ${CURRENT_SOURCE_DIR}/RequestedTypes.h
        ${CURRENT_SOURCE_DIR}/RequestedRenderers.h
        ${CURRENT_SOURCE_DIR}/InstantiationMacros.h
        ${CURRENT_SOURCE_DIR}/Tracer.h
        ${CURRENT_SOURCE_DIR}/Tracer.cu)

    set(SRC_ALL
        ${SRC_RENDERERS_PATH_TRACER}
        ${SRC_COMMON})

    source_group("Renderers/PathTracer" FILES ${SRC_RENDERERS_PATH_TRACER})
    source_group("" FILES ${SRC_COMMON})

    # Target Generation
    gen_device_target(NAME ${GEN_TRACER_DLL_NAME}
                      MACRO ${GEN_TRACER_DLL_MACRO})
    gen_tracer_target(NAME ${GEN_TRACER_DLL_NAME}
                      MACRO ${GEN_TRACER_DLL_MACRO})

    # Kernel and Type file generation
    # Generate the file names
    # TODO: Max thread is overkill but how to set programatically?
    # Create a cache variable and mark it as advanced later
    # cmake_host_system_information(RESULT NUM_CORES
    #                               QUERY NUMBER_OF_LOGICAL_CORES)
    set(NUM_CORES 8)
    tracer_kernel_gen_file_names(NUM_FILES ${NUM_CORES})
    list(TRANSFORM TRACER_KERNEL_GEN_HEADERS PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)
    list(TRANSFORM TRACER_KERNEL_GEN_INSTANTIATIONS PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)
    source_group("" FILES ${TRACER_KERNEL_GEN_HEADERS})
    source_group("Kernel Instantiations" FILES ${TRACER_KERNEL_GEN_INSTANTIATIONS})
    set(SRC_ALL ${SRC_ALL}
        ${TRACER_KERNEL_GEN_HEADERS}
        ${TRACER_KERNEL_GEN_INSTANTIATIONS})

    set(GEN_TARGET_NAME TracerTnKGen_${GEN_TRACER_DLL_NAME})
    add_custom_command(OUTPUT
                       ${TRACER_KERNEL_GEN_HEADERS}
                       ${TRACER_KERNEL_GEN_INSTANTIATIONS}
                       COMMAND TracerKernelGen
                            ${CURRENT_SOURCE_DIR}/TracerTypeGenInput.txt
                            ${NUM_CORES}
                            $<BOOL:${GEN_TRACER_DLL_ENABLE_HW_ACCEL}>
                            ${CMAKE_CURRENT_BINARY_DIR}
                            ${GEN_TRACER_DLL_MACRO}
                            ${GEN_TRACER_DLL_NAME}
                       COMMENT "Generating Kernel Instantiations and Types..."
                       MAIN_DEPENDENCY "${CURRENT_SOURCE_DIR}/TracerTypeGenInput.txt"
                       DEPENDS TracerKernelGen
                       VERBATIM)

    add_custom_target(${GEN_TARGET_NAME}
                      DEPENDS
                      ${TRACER_KERNEL_GEN_HEADERS}
                      ${TRACER_KERNEL_GEN_INSTANTIATIONS})

    # Set *.cu files to compile as either *.hip files
    # or *.cpp files depending on the backend
    change_device_source_file_language(
        MACRO
        ${GEN_TRACER_DLL_MACRO}
        SOURCE_FILES
        ${CURRENT_SOURCE_DIR}/PathTracerRenderer.cu
        ${CURRENT_SOURCE_DIR}/Tracer.cu
        ${TRACER_KERNEL_GEN_INSTANTIATIONS}
    )

    # Actual DLL
    set(TARGET_NAME TracerDLL_${GEN_TRACER_DLL_NAME})
    add_library(${TARGET_NAME} SHARED ${SRC_ALL})

    target_compile_definitions(${TARGET_NAME}
                               PRIVATE
                               MRAY_TRACER_DEVICE_SHARED_EXPORT)

    target_link_libraries(${TARGET_NAME}
                          PRIVATE
                          ${GEN_TRACER_TARGET_NAME}
                          CoreLib
                          TransientPool
                          mray::meta_compile_opts)

    set_target_properties(${TARGET_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON)

    if(GEN_TRACER_DLL_MACRO STREQUAL "MRAY_GPU_BACKEND_CUDA")

        target_link_libraries(${TARGET_NAME} PRIVATE
                              mray::cuda_extra_compile_opts)

        set_target_properties(${TARGET_NAME} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()

    target_include_directories(${TARGET_NAME}
                               PRIVATE
                               ${CMAKE_CURRENT_BINARY_DIR}
                               # For generated files
                               ${CURRENT_SOURCE_DIR})

    add_dependencies(${TARGET_NAME} ${GEN_TARGET_NAME})
    add_precompiled_headers(TARGET ${TARGET_NAME})

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER ${GEN_TRACER_DLL_NAME})
    set_target_properties(${GEN_TARGET_NAME} PROPERTIES FOLDER ${GEN_TRACER_DLL_NAME})
    set_target_properties(${GEN_TRACER_TARGET_NAME} PROPERTIES FOLDER ${GEN_TRACER_DLL_NAME})
    set_target_properties(${GEN_DEVICE_TARGET_NAME} PROPERTIES FOLDER ${GEN_TRACER_DLL_NAME})

endfunction()