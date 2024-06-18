set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Visor)
set(SHADER_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Visor/Shaders)

include("SlangCompileSpirV.cmake")

set(SRC_GUI
    ${CURRENT_SOURCE_DIR}/InputChecker.h
    ${CURRENT_SOURCE_DIR}/MovementSchemes.h
    ${CURRENT_SOURCE_DIR}/VisorGUI.cpp
    ${CURRENT_SOURCE_DIR}/VisorGUI.h)

set(SRC_VISOR
    ${CURRENT_SOURCE_DIR}/FramePool.cpp
    ${CURRENT_SOURCE_DIR}/FramePool.h
    ${CURRENT_SOURCE_DIR}/VisorState.h
    ${CURRENT_SOURCE_DIR}/VisorWindow.cpp
    ${CURRENT_SOURCE_DIR}/VisorWindow.h
    ${CURRENT_SOURCE_DIR}/RenderImagePool.cpp
    ${CURRENT_SOURCE_DIR}/RenderImagePool.h
    ${CURRENT_SOURCE_DIR}/MainUniformBuffer.cpp
    ${CURRENT_SOURCE_DIR}/MainUniformBuffer.h
    ${CURRENT_SOURCE_DIR}/TonemapStage.cpp
    ${CURRENT_SOURCE_DIR}/TonemapStage.h
    ${CURRENT_SOURCE_DIR}/AccumImageStage.cpp
    ${CURRENT_SOURCE_DIR}/AccumImageStage.h
    ${CURRENT_SOURCE_DIR}/FontAtlas.h
    ${CURRENT_SOURCE_DIR}/FontAtlas.cpp)

set(SRC_VULKAN
    ${CURRENT_SOURCE_DIR}/VulkanPipeline.h
    ${CURRENT_SOURCE_DIR}/VulkanPipeline.cpp
    ${CURRENT_SOURCE_DIR}/VulkanTypes.cpp
    ${CURRENT_SOURCE_DIR}/VulkanTypes.h
    ${CURRENT_SOURCE_DIR}/VulkanAllocators.cpp
    ${CURRENT_SOURCE_DIR}/VulkanAllocators.h
    ${CURRENT_SOURCE_DIR}/VulkanCapabilityFinder.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/Visor.cpp
    ${CURRENT_SOURCE_DIR}/Visor.h
    ${CURRENT_SOURCE_DIR}/VisorI.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h)

set(SRC_SHADERS
    ${SHADER_SOURCE_DIR}/AccumInput.slang
    ${SHADER_SOURCE_DIR}/ColorFunctions.slang
    ${SHADER_SOURCE_DIR}/TonemapFunctions.slang
    ${SHADER_SOURCE_DIR}/Tonemap.slang
    ${SHADER_SOURCE_DIR}/TonemapTypeGen.slang)

source_group("Shaders" FILES ${SRC_SHADERS})
source_group("GUI" FILES ${SRC_GUI})
source_group("Vulkan" FILES ${SRC_VULKAN})
source_group("Visor" FILES ${SRC_VISOR})
source_group("" FILES ${SRC_COMMON})

set(SRC_ALL
    ${SRC_GUI}
    ${SRC_SHADERS}
    ${SRC_VISOR}
    ${SRC_VULKAN}
    ${SRC_COMMON})

add_library(Visor SHARED ${SRC_ALL})

target_link_libraries(Visor PRIVATE
                      CoreLib
                      TransientPool
                      ImageLoader
                      glfw3::glfw3
                      imgui::imgui
                      vulkan::vulkan
                      mray::meta_compile_opts)

target_compile_definitions(Visor
                           PRIVATE
                           MRAY_VISOR_SHARED_EXPORT)

add_precompiled_headers(TARGET Visor)

add_custom_command(TARGET Visor POST_BUILD
                   COMMENT "Copying fonts.."
                   COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                   "${MRAY_RESOURCE_DIRECTORY}/Fonts"
                   "${MRAY_CONFIG_BIN_DIRECTORY}/Fonts")

if(MSVC)
add_custom_command(TARGET Visor POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   "${MRAY_CONFIG_LIB_DIRECTORY}/vulkan-1$<$<CONFIG:Debug>:d>.dll"
                   "${MRAY_CONFIG_LIB_DIRECTORY}/glfw3$<$<CONFIG:Debug>:d>.dll"
                   ${MRAY_CONFIG_BIN_DIRECTORY})
endif()

# ==============================#
#        Shader Related         #
# ==============================#
# TODO: make this a cool function later...
set(MRAY_SLANG_COMPILER ${MRAY_PLATFORM_LIB_DIRECTORY}/slang/slangc${CMAKE_EXECUTABLE_SUFFIX})
set(MRAY_SHADER_OUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY}/Shaders)
set(MRAY_SHADER_MODULE_OUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Shaders/$<CONFIG>)
set(SLANG_GEN_SPIRV_OUTPUT_LIST "")

# ============ #
#    MODULES   #
# ============ #
slang_gen_module(OUTPUT "Tonemap"
                 SOURCES
                 ${SHADER_SOURCE_DIR}/Tonemap.slang
                 DEPENDS
                 ${SHADER_SOURCE_DIR}/TonemapFunctions.slang)
slang_gen_module(OUTPUT "ColorFunctions"
                 SOURCES
                 ${SHADER_SOURCE_DIR}/ColorFunctions.slang)

slang_gen_module(OUTPUT "AccumInput"
                 SOURCES
                 ${SHADER_SOURCE_DIR}/AccumInput.slang)

# ============ #
#    SPIR-V    #
# ============ #
slang_gen_spriv(OUTPUT_PREFIX "Tonemap"
                TYPEGEN_SHADER_FILE  "${SHADER_SOURCE_DIR}/TonemapTypeGen.slang"
                TYPEGEN_MACRO "MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB"
                INCLUDES ${SHADER_SOURCE_DIR}
                MODULES "Tonemap" "ColorFunctions"
                GEN_ASSEMBLY)

slang_gen_spriv(OUTPUT_PREFIX "Tonemap"
                TYPEGEN_SHADER_FILE  "${SHADER_SOURCE_DIR}/TonemapTypeGen.slang"
                TYPEGEN_MACRO "MRAY_TONEMAP_EMPTY_ACES_CG_TO_HDR10"
                INCLUDES ${SHADER_SOURCE_DIR}
                MODULES "Tonemap" "ColorFunctions"
                GEN_ASSEMBLY)

slang_gen_spriv(OUTPUT_PREFIX "AccumInput"
                INCLUDES ${SHADER_SOURCE_DIR}
                MODULES "AccumInput"
                GEN_ASSEMBLY)

add_custom_target(VisorShaders DEPENDS ${SLANG_GEN_SPIRV_OUTPUT_LIST})
add_dependencies(Visor VisorShaders)

