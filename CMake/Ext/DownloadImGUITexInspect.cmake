

mray_build_ext_dependency_git(
    NAME imgui_tex_inspect_ext
    URL "https://github.com/yalcinerbora/imgui_tex_inspect.git"
    TAG "d2a2ec1d06f01c4e0bec687a8601ed688390e2ab" # no version, modified for this proj
    LICENSE_NAME "LICENSE.txt"
    BUILD_ARGS
        -DCMAKE_MODULE_PATH=${MRAY_CONFIG_LIB_DIRECTORY}/cmake
    DEPENDS
        imgui_ext
        glbinding_ext

)

ExternalProject_Get_property(imgui_tex_inspect_ext SOURCE_DIR)
set(MRAY_IMGUITI_SRC_LOCATION ${SOURCE_DIR})

# Inject an custom cmakelists.txt and make cmake to use it
ExternalProject_Add_Step(imgui_tex_inspect_ext inject_cmake
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists-ImGUITexInspect.cmake
                        ${MRAY_IMGUITI_SRC_LOCATION}/CMakeLists.txt
                DEPENDEES download update patch
                DEPENDERS configure
                COMMENT "Injecting a cmake lists to imgui_tex_inspect"
)