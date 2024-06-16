function(gen_tracer_target)

    # Parse Args
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_TRACER_TARGET "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tracer)

    set(SRC_TEXTURE
        ${CURRENT_SOURCE_DIR}/TextureMemory.h
        ${CURRENT_SOURCE_DIR}/TextureMemory.cpp)

    set(SRC_PRIMITIVES
        ${CURRENT_SOURCE_DIR}/PrimitiveC.h
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.h
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.hpp
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.cu)

    set(SRC_TRANSFORMS
        ${CURRENT_SOURCE_DIR}/TransformC.h
        ${CURRENT_SOURCE_DIR}/TransformsDefault.h
        ${CURRENT_SOURCE_DIR}/TransformsDefault.hpp
        ${CURRENT_SOURCE_DIR}/TransformsDefault.cu)

    set(SRC_MATERIALS
        ${CURRENT_SOURCE_DIR}/MaterialC.h
        ${CURRENT_SOURCE_DIR}/MaterialsDefault.h
        ${CURRENT_SOURCE_DIR}/MaterialsDefault.hpp
        ${CURRENT_SOURCE_DIR}/MaterialsDefault.cpp)

    set(SRC_CAMERAS
        ${CURRENT_SOURCE_DIR}/CameraC.h
        ${CURRENT_SOURCE_DIR}/CamerasDefault.h
        ${CURRENT_SOURCE_DIR}/CamerasDefault.hpp
        ${CURRENT_SOURCE_DIR}/CamerasDefault.cpp)

    set(SRC_MEDIUMS
        ${CURRENT_SOURCE_DIR}/MediumC.h
        ${CURRENT_SOURCE_DIR}/MediumsDefault.h
        ${CURRENT_SOURCE_DIR}/MediumsDefault.hpp
        ${CURRENT_SOURCE_DIR}/MediumsDefault.cpp)

    set(SRC_LIGHTS
        ${CURRENT_SOURCE_DIR}/LightC.h
        ${CURRENT_SOURCE_DIR}/LightsDefault.h
        ${CURRENT_SOURCE_DIR}/LightsDefault.hpp
        ${CURRENT_SOURCE_DIR}/LightsDefault.cu
        ${CURRENT_SOURCE_DIR}/MetaLight.h
        ${CURRENT_SOURCE_DIR}/MetaLight.hpp)

    set(SRC_ACCELLERATORS
        ${CURRENT_SOURCE_DIR}/AcceleratorC.h
        ${CURRENT_SOURCE_DIR}/AcceleratorWork.h
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.h
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.hpp)

    set(SRC_RENDERERS
        ${CURRENT_SOURCE_DIR}/RendererC.h
        ${CURRENT_SOURCE_DIR}/RenderImage.cpp
        ${CURRENT_SOURCE_DIR}/RenderImage.h
        ${CURRENT_SOURCE_DIR}/ImageRenderer.h
        ${CURRENT_SOURCE_DIR}/ImageRenderer.hpp
        ${CURRENT_SOURCE_DIR}/ImageRenderer.cu)

    set(SRC_RANDOM
        ${CURRENT_SOURCE_DIR}/Random.h
        ${CURRENT_SOURCE_DIR}/Distributions.h
        ${CURRENT_SOURCE_DIR}/Distributions.cu)

    set(SRC_UTILITY
        ${CURRENT_SOURCE_DIR}/RayPartitioner.h
        ${CURRENT_SOURCE_DIR}/RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/ParamVaryingData.h
        ${CURRENT_SOURCE_DIR}/ShapeFunctions.h
        ${CURRENT_SOURCE_DIR}/DistributionFunctions.h
        ${CURRENT_SOURCE_DIR}/Key.h
        ${CURRENT_SOURCE_DIR}/KeyFormat.h
        ${CURRENT_SOURCE_DIR}/Hit.h)

    set(SRC_COMMON
        ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
        ${CURRENT_SOURCE_DIR}/TracerTypes.h
        ${CURRENT_SOURCE_DIR}/GenericGroup.h
        ${CURRENT_SOURCE_DIR}/TracerBase.h
        ${CURRENT_SOURCE_DIR}/TracerBase.cpp)

    set(SRC_ALL
        ${SRC_TEXTURE}
        ${SRC_PRIMITIVES}
        ${SRC_MATERIALS}
        ${SRC_CAMERAS}
        ${SRC_MEDIUMS}
        ${SRC_LIGHTS}
        ${SRC_TRANSFORMS}
        ${SRC_ACCELLERATORS}
        ${SRC_RENDERERS}
        ${SRC_RANDOM}
        ${SRC_UTILITY}
        ${SRC_COMMON})

    # IDE Filters
    source_group("Texture" FILES ${SRC_TEXTURE})
    source_group("Primitives" FILES ${SRC_PRIMITIVES})
    source_group("Transforms" FILES ${SRC_TRANSFORMS})
    source_group("Materials" FILES ${SRC_MATERIALS})
    source_group("Mediums" FILES ${SRC_MEDIUMS})
    source_group("Cameras" FILES ${SRC_CAMERAS})
    source_group("Accelerators" FILES ${SRC_ACCELLERATORS})
    source_group("Lights" FILES ${SRC_LIGHTS})
    source_group("Random" FILES ${SRC_RANDOM})
    source_group("Renderers" FILES ${SRC_RENDERERS})
    source_group("Utility" FILES ${SRC_UTILITY})
    source_group("" FILES ${SRC_COMMON})

    set(TARGET_FULL_NAME "Tracer${GEN_TRACER_TARGET_NAME}")
    set(DEVICE_TARGET_FULL_NAME "Device${GEN_TRACER_TARGET_NAME}")
    add_library(${TARGET_FULL_NAME} STATIC ${SRC_ALL})

    target_link_libraries(${TARGET_FULL_NAME}
                          PUBLIC
                          ${DEVICE_TARGET_FULL_NAME}
                          CoreLib
                          TransientPool
                          bs::thread_pool
                          PRIVATE
                          mray::meta_compile_opts
                          mray::cuda_extra_compile_opts)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION ON
                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})

    set(GEN_TRACER_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)

endfunction()