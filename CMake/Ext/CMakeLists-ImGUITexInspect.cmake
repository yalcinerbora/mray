cmake_minimum_required(VERSION 3.25)

include(GNUInstallDirs)

project(ImguiTexInspectInject LANGUAGES CXX)

find_package(imgui REQUIRED)

add_library(imgui_tex_inspect STATIC)

# TODO: Can we neatly install this?
# An idea inject a CMakeLists.txt
# to dearimgui (add as a before configure step thing)
# then compile and run

target_sources(imgui_tex_inspect PUBLIC FILE_SET HEADERS
               BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} # This is correct?
               FILES
               # Headers
               ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect_internal.h
               ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect.h
)

target_sources(imgui_tex_inspect PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect_demo.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/backends/tex_inspect_opengl.cpp
)

# imgui backends directly include imgui.h
# so we cannot add backends to backend folder directly install these
set(IMGUITI_BACKEND_HEADERS
    # Backend Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/tex_inspect_opengl.h
)

target_link_libraries(imgui_tex_inspect imgui)

target_compile_definitions(imgui_tex_inspect PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)

target_include_directories(imgui_tex_inspect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Lib/Include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Lib/Include/glbinding/3rdparty)

install(TARGETS imgui_tex_inspect
        EXPORT imgui_tex_inspectTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Imgui)

install(FILES  ${IMGUITI_BACKEND_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Imgui)

# From the cmake docs tutorial
include(CMakePackageConfigHelpers)

set(IMGUITI_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/imgui_tex_inspect)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspectConfig.cmake.in
     " @PACKAGE_INIT@
include (\"\$\{CMAKE_CURRENT_LIST_DIR\}/imgui_tex_inspectTargets.cmake\")")

# generate the config file that includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspectConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/imgui_tex_inspectConfig.cmake"
        INSTALL_DESTINATION ${IMGUITI_CONFIG_DIR}
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/imgui_tex_inspectConfig.cmake"
        DESTINATION ${IMGUITI_CONFIG_DIR})

install(EXPORT imgui_tex_inspectTargets
        FILE imgui_tex_inspectTargets.cmake
        DESTINATION ${IMGUITI_CONFIG_DIR})
