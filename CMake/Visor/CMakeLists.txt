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