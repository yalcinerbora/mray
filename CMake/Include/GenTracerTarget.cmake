function(gen_tracer_target)

    # Parse Args
    set(options)
    set(oneValueArgs NAME MACRO)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_TRACER_TARGET "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    # OptiX Check (Backend: CUDA, Enable HW Acceleration ON)
    if(MRAY_ENABLE_HW_ACCELERATION AND GEN_TRACER_TARGET_NAME STREQUAL "CUDA")
       set(MRAY_TRACER_OPTIX ON)
    else()
        set(MRAY_TRACER_OPTIX OFF)
    endif()
    # Embree Check (Backend: CPU, Enable HW Acceleration ON)
    if(MRAY_ENABLE_HW_ACCELERATION AND GEN_TRACER_TARGET_NAME STREQUAL "CPU")
       set(MRAY_TRACER_EMBREE ON)
    else()
        set(MRAY_TRACER_EMBREE OFF)
    endif()
    # TODO: HIP RT Check (Backend: HIP, Enable HW Acceleration ON)

    set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tracer)
    set(SRC_TEXTURE
        ${CURRENT_SOURCE_DIR}/BCColorIO.h
        ${CURRENT_SOURCE_DIR}/TextureMemory.h
        ${CURRENT_SOURCE_DIR}/TextureMemory.cpp
        ${CURRENT_SOURCE_DIR}/StreamingTextureCache.h
        ${CURRENT_SOURCE_DIR}/StreamingTextureCache.cpp
        ${CURRENT_SOURCE_DIR}/StreamingTextureView.h
        ${CURRENT_SOURCE_DIR}/StreamingTextureView.hpp
        ${CURRENT_SOURCE_DIR}/GenericTexture.cpp
        ${CURRENT_SOURCE_DIR}/GenericTextureRW.h
        ${CURRENT_SOURCE_DIR}/ColorConverter.h
        ${CURRENT_SOURCE_DIR}/ColorConverter.cu
        ${CURRENT_SOURCE_DIR}/TextureCommon.h
        ${CURRENT_SOURCE_DIR}/TextureFilter.h
        ${CURRENT_SOURCE_DIR}/TextureFilter.cu
        ${CURRENT_SOURCE_DIR}/TextureView.h
        ${CURRENT_SOURCE_DIR}/TextureView.hpp
        ${CURRENT_SOURCE_DIR}/Texture.h)

    set(SRC_PRIMITIVES
        ${CURRENT_SOURCE_DIR}/PrimitiveC.h
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.h
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.hpp
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.cu
        ${CURRENT_SOURCE_DIR}/PrimitivesDefault.h
        ${CURRENT_SOURCE_DIR}/PrimitivesDefault.hpp
        ${CURRENT_SOURCE_DIR}/PrimitivesDefault.cu)

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
        ${CURRENT_SOURCE_DIR}/AcceleratorCommon.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorWork.h
        ${CURRENT_SOURCE_DIR}/AcceleratorWork.kt.h
        ${CURRENT_SOURCE_DIR}/AcceleratorWorkI.h
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.h
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.hpp
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.h
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.hpp)

    set(SRC_RENDERERS
        ${CURRENT_SOURCE_DIR}/RenderWork.h
        ${CURRENT_SOURCE_DIR}/RenderWork.kt.h
        ${CURRENT_SOURCE_DIR}/RendererC.h
        ${CURRENT_SOURCE_DIR}/RendererCommon.h
        ${CURRENT_SOURCE_DIR}/RendererCommon.cu
        ${CURRENT_SOURCE_DIR}/PathTracerRendererBase.h
        ${CURRENT_SOURCE_DIR}/PathTracerRendererBase.cu
        ${CURRENT_SOURCE_DIR}/RenderImage.cpp
        ${CURRENT_SOURCE_DIR}/RenderImage.h
        ${CURRENT_SOURCE_DIR}/RayGenKernels.h
        ${CURRENT_SOURCE_DIR}/RayGenKernels.kt.h
        ${CURRENT_SOURCE_DIR}/LightSampler.hpp
        ${CURRENT_SOURCE_DIR}/LightSampler.h)

    set(SRC_RENDERERS_TEX_VIEW
        ${CURRENT_SOURCE_DIR}/TexViewRenderer.h
        ${CURRENT_SOURCE_DIR}/TexViewRenderer.cu)

    set(SRC_RENDERERS_SURFACE
        ${CURRENT_SOURCE_DIR}/SurfaceRendererShaders.h
        ${CURRENT_SOURCE_DIR}/SurfaceRenderer.h
        ${CURRENT_SOURCE_DIR}/SurfaceRenderer.cu)

    set(SRC_RENDERERS_HASH_GRID
        ${CURRENT_SOURCE_DIR}/HashGridRendererShaders.h
        ${CURRENT_SOURCE_DIR}/HashGridRenderer.h
        ${CURRENT_SOURCE_DIR}/HashGridRenderer.cu)

    set(SRC_RANDOM
        ${CURRENT_SOURCE_DIR}/Random.cu
        ${CURRENT_SOURCE_DIR}/Random.h
        ${CURRENT_SOURCE_DIR}/DistributionFunctions.h
        ${CURRENT_SOURCE_DIR}/Distributions.h
        ${CURRENT_SOURCE_DIR}/Distributions.cu
        ${CURRENT_SOURCE_DIR}/SobolMatrices.cpp
        ${CURRENT_SOURCE_DIR}/SobolMatrices.h)

    set(SRC_SPECTRUM
        ${CURRENT_SOURCE_DIR}/SpectrumContext.cu
        ${CURRENT_SOURCE_DIR}/SpectrumContext.hpp
        ${CURRENT_SOURCE_DIR}/SpectrumContext.h
        ${CURRENT_SOURCE_DIR}/SpectrumC.h)

    set(SRC_UTILITY
        ${CURRENT_SOURCE_DIR}/Bitspan.h
        ${CURRENT_SOURCE_DIR}/RayPartitioner.h
        ${CURRENT_SOURCE_DIR}/RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/ParamVaryingData.h
        ${CURRENT_SOURCE_DIR}/Key.h
        ${CURRENT_SOURCE_DIR}/TypeFormat.h
        ${CURRENT_SOURCE_DIR}/Hit.h
        ${CURRENT_SOURCE_DIR}/Filters.h
        ${CURRENT_SOURCE_DIR}/SurfaceComparators.h
        ${CURRENT_SOURCE_DIR}/HashGrid.h
        ${CURRENT_SOURCE_DIR}/HashGrid.cu
        ${CURRENT_SOURCE_DIR}/MediaTracker.h
        ${CURRENT_SOURCE_DIR}/MediaTracker.cu)

    set(SRC_COMMON
        ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
        ${CURRENT_SOURCE_DIR}/TracerTypes.h
        ${CURRENT_SOURCE_DIR}/GenericGroup.cpp
        ${CURRENT_SOURCE_DIR}/GenericGroup.hpp
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
        ${SRC_RENDERERS_HASH_GRID}
        ${SRC_RANDOM}
        ${SRC_SPECTRUM}
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
    source_group("Renderers/HashGrid" FILES ${SRC_RENDERERS_HASH_GRID})
    source_group("Spectrum" FILES ${SRC_SPECTRUM})
    source_group("Utility" FILES ${SRC_UTILITY})
    source_group("" FILES ${SRC_COMMON})

    change_device_source_file_language(
        MACRO
        ${GEN_TRACER_TARGET_MACRO}
        SOURCE_FILES
        ${CURRENT_SOURCE_DIR}/ColorConverter.cu
        ${CURRENT_SOURCE_DIR}/TextureFilter.cu
        ${CURRENT_SOURCE_DIR}/PrimitiveDefaultTriangle.cu
        ${CURRENT_SOURCE_DIR}/PrimitivesDefault.cu
        ${CURRENT_SOURCE_DIR}/TransformsDefault.cu
        ${CURRENT_SOURCE_DIR}/LightsDefault.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorCommon.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorLinear.cu
        ${CURRENT_SOURCE_DIR}/AcceleratorLBVH.cu
        ${CURRENT_SOURCE_DIR}/TexViewRenderer.cu
        ${CURRENT_SOURCE_DIR}/SurfaceRenderer.cu
        ${CURRENT_SOURCE_DIR}/HashGridRenderer.cu
        ${CURRENT_SOURCE_DIR}/Random.cu
        ${CURRENT_SOURCE_DIR}/Distributions.cu
        ${CURRENT_SOURCE_DIR}/RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/SpectrumContext.cu
        ${CURRENT_SOURCE_DIR}/HashGrid.cu
        ${CURRENT_SOURCE_DIR}/RendererCommon.cu
        ${CURRENT_SOURCE_DIR}/PathTracerRendererBase.cu
        ${CURRENT_SOURCE_DIR}/MediaTracker.cu
    )

    # Add sources for OptiX (Backend: CUDA, Enable HW Acceleration ON)
    if(MRAY_TRACER_OPTIX)

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

    # Add sources for Embree (Backend: CPU, Enable HW Acceleration ON)
    if(MRAY_TRACER_EMBREE)
        set(SRC_ACCELLERATORS_HW
            ${CURRENT_SOURCE_DIR}/Embree/AcceleratorEmbree.cpp
            ${CURRENT_SOURCE_DIR}/Embree/AcceleratorEmbree.h
            ${CURRENT_SOURCE_DIR}/Embree/AcceleratorEmbree.hpp)

        set(SRC_ALL ${SRC_ALL} ${SRC_ACCELLERATORS_HW})

        source_group("Accelerators/Embree" FILES
                     ${SRC_ACCELLERATORS_HW})
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
                          PRIVATE
                          mray::meta_compile_opts)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON)

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})



    if(MRAY_ENABLE_HW_ACCELERATION)
        target_compile_definitions(${TARGET_FULL_NAME}
                                   PUBLIC
                                   MRAY_ENABLE_HW_ACCELERATION)
        # Add current source dir as include
        # so that the OptiX/Embree/HIPRT folder can access the AcceleratorC.h etc.
        target_include_directories(${TARGET_FULL_NAME}
                                   PUBLIC
                                   ${CURRENT_SOURCE_DIR})
    endif()

    # Optix Related Target definitions
    if(MRAY_TRACER_OPTIX)
        # Check if user
        option(MRAY_COMPILE_OPTIX_AS_PTX "Compile ptx for optix instead of optixir" OFF)
        mark_as_advanced(MRAY_COMPILE_OPTIX_AS_PTX)

        target_link_libraries(${TARGET_FULL_NAME}
                              PRIVATE
                              spdlog::spdlog
                              PUBLIC
                              optix::optix)

        if(MRAY_COMPILE_OPTIX_AS_PTX)
            target_compile_definitions(${TARGET_FULL_NAME}
                                       PRIVATE
                                       MRAY_COMPILE_OPTIX_AS_PTX)
        endif()

        add_custom_command(TARGET ${TARGET_FULL_NAME} POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           "$<TARGET_FILE:spdlog::spdlog>"
                           ${MRAY_CONFIG_BIN_DIRECTORY})

        # ======================================= #
        # Generate OptiX-IR compilation target    #
        # ======================================= #
        # Here we can not directly use the nvcc_compile_optix_ir
        # since we set "set_source_files_properties" to HEADER_FILE_ONLY
        # to be able to see the sources from the "Tracer_CUDA.lib" target.
        # This will creep to the nvcc_compile_optix_ir and it won't compile
        # the files. So we need to create a scope. We do this with
        # "add_subdirectory". (In hindsight, there is direct scope parameter
        # in CMake i think, this is little bit more readable though).
        #
        # Moreover, a little bit of spaghetti, since this is a function
        # so the include dir must be present on the caller's side.
        # So the folder "OptiX" should be in "TracerDevice" folder
        # instead of the "Include" folder.
        add_subdirectory(OptiX)
    endif()
    # Add Embree4 link
    if(MRAY_TRACER_EMBREE)
        target_link_libraries(${TARGET_FULL_NAME}
                              PUBLIC
                              embree4::embree4_cpu)

        # Copy DLLS
        if(WIN32)
            add_custom_command(TARGET ${TARGET_FULL_NAME} POST_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy_if_different
                               "$<TARGET_FILE:embree4::embree4_cpu>"
                               # We need this as well....
                               # This means in debug builds, we will have
                               # two TBB one debug one release
                               # I've tried to fix this but could not do it properly
                               "$<$<CONFIG:Debug>:${MRAY_PLATFORM_LIB_DIRECTORY}/embree/lib/tbb12.dll>"
                               ${MRAY_CONFIG_BIN_DIRECTORY}
                               COMMENT "Copying Embree DLLs")
        else()
            add_custom_command(TARGET ${TARGET_FULL_NAME} POST_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy_if_different
                               "$<TARGET_FILE:embree4::embree4_cpu>"
                               # Read the comment on the Windows side...
                               "$<$<CONFIG:Debug>:${MRAY_PLATFORM_LIB_DIRECTORY}/Release/libtbb.so.12>"
                               ${MRAY_CONFIG_BIN_DIRECTORY}
                               COMMENT "Copying Embree DLLs")
        endif()


    endif()

    # CUDA-only enable HW seperable compilation and flags
    if(GEN_TRACER_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_CUDA")
        target_link_libraries(${TARGET_FULL_NAME} PRIVATE
                              mray::cuda_extra_compile_opts)

        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    elseif(GEN_TRACER_TARGET_MACRO STREQUAL "MRAY_GPU_BACKEND_CPU")
        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              INTERPROCEDURAL_OPTIMIZATION_RELEASE
                              ${MRAY_HOST_BACKEND_IPO_MODE})
    endif()

    # Return the generated target
    set(GEN_TRACER_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)

endfunction()