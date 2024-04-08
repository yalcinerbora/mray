set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Visor)
set(SHADER_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Visor/Shaders)

include(SlangCompileSpirV.cmake)

set(SRC_WINDOW
    ${CURRENT_SOURCE_DIR}/VisorWindow.cpp
    ${CURRENT_SOURCE_DIR}/VisorWindow.h
    ${CURRENT_SOURCE_DIR}/VisorI.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/VulkanAllocators.cpp
    ${CURRENT_SOURCE_DIR}/VulkanAllocators.h
    ${CURRENT_SOURCE_DIR}/Visor.cpp
    ${CURRENT_SOURCE_DIR}/Visor.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h)

set(SRC_SHADERS
    ${SHADER_SOURCE_DIR}/ColorFunctions.slang
    ${SHADER_SOURCE_DIR}/TonemapFunctions.slang
    ${SHADER_SOURCE_DIR}/Tonemap.slang
    ${SHADER_SOURCE_DIR}/TonemapTypeGen.slang)

source_group("Shaders" FILES ${SRC_SHADERS})
source_group("Window" FILES ${SRC_WINDOW})
source_group("" FILES ${SRC_COMMON})

set(SRC_ALL
    ${SRC_SHADERS}
    ${SRC_WINDOW}
    ${SRC_COMMON})

add_library(Visor SHARED ${SRC_ALL})

target_link_libraries(Visor PRIVATE
                      CoreLib
                      glfw3::glfw3
                      imgui::imgui
                      vulkan::vulkan
                      mray::meta_compile_opts)

target_compile_definitions(Visor
                           PRIVATE
                           MRAY_VISOR_SHARED_EXPORT)

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
                 ${SHADER_SOURCE_DIR}/Tonemap.slang)
slang_gen_module(OUTPUT "ColorFunctions"
                 SOURCES
                 ${SHADER_SOURCE_DIR}/ColorFunctions.slang)

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

add_custom_target(VisorShaders DEPENDS ${SLANG_GEN_SPIRV_OUTPUT_LIST})
add_dependencies(Visor VisorShaders)
