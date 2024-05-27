cmake_minimum_required(VERSION 3.25)

include(GNUInstallDirs)

project(ImguiInject LANGUAGES CXX)

find_package(glfw3 REQUIRED)

add_library(imgui STATIC)

# TODO: Can we neatly install this?
# An idea inject a CMakeLists.txt
# to dearimgui (add as a before configure step thing)
# then compile and run

target_sources(imgui PUBLIC FILE_SET HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} # This is correct?
            FILES
            # Headers
            ${CMAKE_CURRENT_SOURCE_DIR}/imconfig.h
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui.h
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_internal.h
            ${CMAKE_CURRENT_SOURCE_DIR}/imstb_rectpack.h
            ${CMAKE_CURRENT_SOURCE_DIR}/imstb_textedit.h
            ${CMAKE_CURRENT_SOURCE_DIR}/imstb_truetype.h
)

target_sources(imgui PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_demo.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_draw.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tables.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_widgets.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

# Position independent code is required for static library on Linux (ld?)
set_target_properties(imgui PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

# imgui backends directly include imgui.h
# so we cannot add backends to backend folder directly install these
set(IMGUI_BACKEND_HEADERS
    # Backend Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_glfw.h
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_vulkan.h
)

target_link_libraries(imgui glfw)

install(TARGETS imgui
        EXPORT imguiTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Imgui)

install(FILES  ${IMGUI_BACKEND_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Imgui)

# From the cmake docs tutorial
include(CMakePackageConfigHelpers)

set(IMGUI_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/imgui)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/imguiConfig.cmake.in
     " @PACKAGE_INIT@
include (\"\${CMAKE_CURRENT_LIST_DIR}/imguiTargets.cmake\")")

# generate the config file that includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/imguiConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/imguiConfig.cmake"
        INSTALL_DESTINATION ${IMGUI_CONFIG_DIR}
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/imguiConfig.cmake"
        DESTINATION ${IMGUI_CONFIG_DIR})

install(EXPORT imguiTargets
        FILE imguiTargets.cmake
        DESTINATION ${IMGUI_CONFIG_DIR})