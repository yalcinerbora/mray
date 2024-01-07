set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tracer)

message(STATUS "Configuring ${CURRENT_SOURCE_DIR}")

set(SRC_PRIMITIVES
    ${CURRENT_SOURCE_DIR}/PrimitiveC.h
    ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.h
    ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.hpp)

set(SRC_TRANSFORMS
    ${CURRENT_SOURCE_DIR}/TransformC.h
    ${CURRENT_SOURCE_DIR}/Transforms.h
    ${CURRENT_SOURCE_DIR}/Transforms.hpp)

set(SRC_MATERIALS
    ${CURRENT_SOURCE_DIR}/MaterialC.h
    ${CURRENT_SOURCE_DIR}/MaterialsDefault.h
    ${CURRENT_SOURCE_DIR}/MaterialsDefault.hpp)

set(SRC_RENDERERS
    ${CURRENT_SOURCE_DIR}/RendererC.h)

set(SRC_UTILITY
    ${CURRENT_SOURCE_DIR}/ParamVaryingData.h
    ${CURRENT_SOURCE_DIR}/ShapeFunctions.h
    ${CURRENT_SOURCE_DIR}/GraphicsFunctions.h
    ${CURRENT_SOURCE_DIR}/Key.h
    ${CURRENT_SOURCE_DIR}/Hit.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/TracerTypes.h)

set(SRC_ALL
    ${SRC_PRIMITIVES}
    ${SRC_MATERIALS}
    ${SRC_TRANSFORMS}
    ${SRC_RENDERERS}
    ${SRC_UTILITY}
    ${SRC_COMMON})

# IDE Filters
source_group("Primitives" FILES ${SRC_PRIMITIVES})
source_group("Transforms" FILES ${SRC_TRANSFORMS})
source_group("Materials" FILES ${SRC_MATERIALS})
source_group("Renderers" FILES ${SRC_RENDERERS})
source_group("Utility" FILES ${SRC_UTILITY})
source_group("" FILES ${SRC_COMMON})

add_library(TracerCUDA SHARED ${SRC_ALL})

target_link_libraries(TracerCUDA PRIVATE
                      DeviceLib
                      CoreLib
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts)

set_target_properties(TracerCUDA PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)