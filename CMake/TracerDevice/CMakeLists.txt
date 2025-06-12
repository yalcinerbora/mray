
include("GenDeviceTarget")
include("GenTracerTarget")
include("GenTracerTypesAndKernels")

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

# I dunno why but global set did not work
if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_HIP")

    set_source_files_properties(${CURRENT_SOURCE_DIR}/PathTracerRenderer.cu
                                ${CURRENT_SOURCE_DIR}/Tracer.cu
                                ${CURRENT_SOURCE_DIR}/Renderers_0.cu
                                ${CURRENT_SOURCE_DIR}/Renderers_1.cu
                                ${CURRENT_SOURCE_DIR}/Renderers_2.cu
                                PROPERTIES LANGUAGE HIP)

endif()

# Find OptiX if requested
if(MRAY_ENABLE_HW_ACCELERATION)
    find_package(OPTIX 8)
    if(NOT OPTIX_FOUND)
        # Maybe we could not able to find the optix
        # expose the OPTIX_DIR (in cmake-GUI) so user can manually set it
        message(FATAL_ERROR "OptiX Not Found, either set OPTIX_INSTALL_DIR or disable MRAY_ENABLE_HW_ACCELERATION")
        set(MRAY_ENABLE_HW_ACCELERATION CACHE BOOL OFF)
    endif()
endif()

# Target Generation
gen_device_target(NAME ${MRAY_GPU_PLATFORM_NAME}
                  MACRO ${MRAY_DEVICE_BACKEND})
gen_tracer_target(NAME ${MRAY_GPU_PLATFORM_NAME})

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

set(GEN_TARGET_NAME TracerTnKGen_${MRAY_GPU_PLATFORM_NAME})
add_custom_command(OUTPUT
                   ${TRACER_KERNEL_GEN_HEADERS}
                   ${TRACER_KERNEL_GEN_INSTANTIATIONS}
                   COMMAND TracerKernelGen
                   ${CURRENT_SOURCE_DIR}/TracerTypeGenInput.txt
                   ${NUM_CORES}
                   ${CMAKE_CURRENT_BINARY_DIR}
                   ${MRAY_DEVICE_BACKEND}
                   COMMENT "Generating Kernel Instantiations and Types..."
                   MAIN_DEPENDENCY "${CURRENT_SOURCE_DIR}/TracerTypeGenInput.txt"
                   DEPENDS TracerKernelGen
                   VERBATIM)

add_custom_target(${GEN_TARGET_NAME}
                  DEPENDS
                  ${TRACER_KERNEL_GEN_HEADERS}
                  ${TRACER_KERNEL_GEN_INSTANTIATIONS})

if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_HIP")

    set_source_files_properties(${TRACER_KERNEL_GEN_HEADERS}
                                PROPERTIES LANGUAGE HIP)

endif()

# Actual DLL
set(TARGET_NAME TracerDLL_${MRAY_GPU_PLATFORM_NAME})
add_library(${TARGET_NAME} SHARED ${SRC_ALL})

target_compile_definitions(${TARGET_NAME}
                           PRIVATE
                           MRAY_TRACER_DEVICE_SHARED_EXPORT)

target_link_libraries(${TARGET_NAME}
                      PRIVATE
                      ${GEN_TRACER_TARGET_NAME}
                      CoreLib
                      TransientPool
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts)

target_include_directories(${TARGET_NAME}
                           PRIVATE
                           ${CMAKE_CURRENT_BINARY_DIR}
                           # For generated files
                           ${CURRENT_SOURCE_DIR})

add_dependencies(${TARGET_NAME} ${GEN_TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_precompiled_headers(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})
set_target_properties(${GEN_TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})
set_target_properties(${GEN_TRACER_TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})
set_target_properties(${GEN_DEVICE_TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})
