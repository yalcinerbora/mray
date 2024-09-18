function(gen_tracer_target)

    # Parse Args
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_TRACER_TARGET "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(MRAY_ENABLE_HW_ACCELERATION AND
       GEN_TRACER_TARGET_NAME STREQUAL "CUDA")
       set(MRAY_OPTIX ON)
    else()
        set(MRAY_OPTIX OFF)
    endif()

    set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tracer)

    set(SRC_TEXTURE
        ${CURRENT_SOURCE_DIR}/BCColorIO.h
        ${CURRENT_SOURCE_DIR}/TextureMemory.h
        ${CURRENT_SOURCE_DIR}/TextureMemory.cpp
        ${CURRENT_SOURCE_DIR}/GenericTexture.cpp
        ${CURRENT_SOURCE_DIR}/GenericTextureRW.h
        ${CURRENT_SOURCE_DIR}/ColorConverter.h
        ${CURRENT_SOURCE_DIR}/ColorConverter.cu
        ${CURRENT_SOURCE_DIR}/TextureCommon.h
        ${CURRENT_SOURCE_DIR}/TextureFilter.h
        ${CURRENT_SOURCE_DIR}/TextureFilter.cu
        ${CURRENT_SOURCE_DIR}/TextureView.h)

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
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.hpp
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.h
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.hpp)

    set(SRC_RENDERERS
        ${CURRENT_SOURCE_DIR}/RenderWork.h
        ${CURRENT_SOURCE_DIR}/RendererC.h
        ${CURRENT_SOURCE_DIR}/RenderImage.cpp
        ${CURRENT_SOURCE_DIR}/RenderImage.h
        ${CURRENT_SOURCE_DIR}/RayGenKernels.h)

    set(SRC_RENDERERS_TEX_VIEW
        ${CURRENT_SOURCE_DIR}/TexViewRenderer.h
        ${CURRENT_SOURCE_DIR}/TexViewRenderer.hpp
        ${CURRENT_SOURCE_DIR}/TexViewRenderer.cu)

    set(SRC_RENDERERS_SURFACE
        ${CURRENT_SOURCE_DIR}/SurfaceRenderer.h
        ${CURRENT_SOURCE_DIR}/SurfaceRenderer.hpp
        ${CURRENT_SOURCE_DIR}/SurfaceRenderer.cu)

    set(SRC_RANDOM
        ${CURRENT_SOURCE_DIR}/Random.cu
        ${CURRENT_SOURCE_DIR}/Random.h
        ${CURRENT_SOURCE_DIR}/DistributionFunctions.h
        ${CURRENT_SOURCE_DIR}/Distributions.h
        ${CURRENT_SOURCE_DIR}/Distributions.cu)

    set(SRC_UTILITY
        ${CURRENT_SOURCE_DIR}/RayPartitioner.h
        ${CURRENT_SOURCE_DIR}/RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/ParamVaryingData.h
        ${CURRENT_SOURCE_DIR}/Key.h
        ${CURRENT_SOURCE_DIR}/TypeFormat.h
        ${CURRENT_SOURCE_DIR}/Hit.h
        ${CURRENT_SOURCE_DIR}/Filters.h)

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
        ${SRC_RENDERERS_TEX_VIEW}
        ${SRC_RENDERERS_SURFACE}
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
    source_group("Renderers/Surface" FILES ${SRC_RENDERERS_SURFACE})
    source_group("Renderers/TexView" FILES ${SRC_RENDERERS_TEX_VIEW})
    source_group("Utility" FILES ${SRC_UTILITY})
    source_group("" FILES ${SRC_COMMON})

    # Enable/Disable HW Acceleration
    if(MRAY_OPTIX)

        set(SRC_ACCELLERATORS_HW
            ${CURRENT_SOURCE_DIR}/OptiX/AcceleratorOptiX.cu
            ${CURRENT_SOURCE_DIR}/OptiX/AcceleratorOptiX.h
            ${CURRENT_SOURCE_DIR}/OptiX/AcceleratorOptiX.hpp)

        set(SRC_ACCELLERATORS_PTX
            ${CURRENT_SOURCE_DIR}/OptiX/OptiXPTX.cu
            ${CURRENT_SOURCE_DIR}/OptiX/OptiXPTX.h)

        set(SRC_ALL ${SRC_ALL}
            ${SRC_ACCELLERATORS_HW}
            ${SRC_ACCELLERATORS_PTX})

        # Make OptiX Sources not participate in build
        # only on the Tracer Target (so we can accsess
        # them easily in IDE)
        set_source_files_properties(${SRC_ACCELLERATORS_PTX}
            PROPERTIES HEADER_FILE_ONLY TRUE)

        source_group("Accelerators/OptiX" FILES
                     ${SRC_ACCELLERATORS_HW})
        source_group("Accelerators/OptiX/IR" FILES
                     ${SRC_ACCELLERATORS_PTX})
        source_group("" FILES ${SRC_ACCELLERATORS_PTX})
    endif()

    # Finally Gen Library
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

    # Optix Related Target definitions
    if(MRAY_OPTIX)
        # Add current source dir as include
        # so that the OptiX folder can access the AcceleratorC.h etc.
        target_include_directories(${TARGET_FULL_NAME}
                                   PUBLIC
                                   ${CURRENT_SOURCE_DIR}
                                   ${OPTIX_INCLUDE_DIR})
        target_link_libraries(${TARGET_FULL_NAME}
                              PRIVATE
                              spdlog::spdlog)
        target_compile_definitions(${TARGET_FULL_NAME}
                                   PUBLIC
                                   MRAY_ENABLE_HW_ACCELERATION)
        if(MSVC)
            add_custom_command(TARGET ${TARGET_FULL_NAME} PRE_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${MRAY_CONFIG_LIB_DIRECTORY}/spdlog$<$<CONFIG:Debug>:d>.dll"
                            ${MRAY_CONFIG_BIN_DIRECTORY})
        endif()
    endif()

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION ON
                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})

    set(GEN_TRACER_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)

    # Generate OptiX-IR compilation target
    if(MRAY_OPTIX)
        # Here we can not directly use the nvcc_compile_optix_ir
        # since we set set_source_files_properties to HEADER_FILE_ONLY
        # This will creep up to the nvcc_compile_optix_ir
        # Little bit of spagetti, since this is function itself
        # thus, folder OptiX should be in "TracerDevice" folder
        # instead of the "Include" folder
        add_subdirectory(OptiX)
    endif()

endfunction()