set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Visor)

set(SRC_WINDOW
    ${CURRENT_SOURCE_DIR}/Visor.cpp
    ${CURRENT_SOURCE_DIR}/Visor.h
    ${CURRENT_SOURCE_DIR}/VisorI.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h)

source_group("Window" FILES ${SRC_WINDOW})
source_group("" FILES ${SRC_COMMON})

set(SRC_ALL
    ${SRC_WINDOW}
    ${SRC_COMMON})

add_library(Visor SHARED ${SRC_ALL})

target_link_libraries(Visor PRIVATE
                      CoreLib
                      glfw3::glfw3
                      imgui::imgui
                      vulkan::vulkan-1
                      mray::meta_compile_opts)

target_compile_definitions(Visor
                           PRIVATE
                           MRAY_VISOR_SHARED_EXPORT)