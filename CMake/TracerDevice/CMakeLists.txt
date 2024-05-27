
include("GenDeviceTarget")
include("GenTracerTarget")

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/TracerDLL)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/Tracer.h
    ${CURRENT_SOURCE_DIR}/Tracer.cu)

set(SRC_ALL
    ${SRC_COMMON})

source_group("" FILES ${SRC_COMMON})

# Target Generation
gen_device_target(NAME ${MRAY_GPU_PLATFORM_NAME}
                  MACRO ${MRAY_DEVICE_BACKEND})
gen_tracer_target(NAME ${MRAY_GPU_PLATFORM_NAME})

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

set_target_properties(${TARGET_NAME} PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})
set_target_properties(${GEN_TRACER_TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})
set_target_properties(${GEN_DEVICE_TARGET_NAME} PROPERTIES FOLDER ${MRAY_GPU_PLATFORM_NAME})