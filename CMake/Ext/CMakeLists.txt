cmake_minimum_required(VERSION 3.26)

# CMAKE Project Definitions and Source, Library Folders are separate
# Directories
set(MRAY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
cmake_path(NORMAL_PATH MRAY_DIRECTORY)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MRAY_DIRECTORY}CMake")

# Do not use in source builds
include(PreventInSourceBuilds)

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Configuration types, for multi-config generators (MRay supports only Debug and Release for external libraries)")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Configuration type for single-config generators. (MRay supports only Debug and Release for external libraries)")

# ExternalProject_Add warpper
include(ExtDownloadDeps)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create a project..
project(MRayExternal DESCRIPTION "MRay dependency downloader/compiler")

option(MRAY_BUILD_TESTS "Build Tests" ON)

# Populate the meta directories for this project hierarchy
include(Macros)
generate_platform_name()
populate_mray_directory_variables(${MRAY_DIRECTORY})

# External Dependencies
# Comparably Large dependencies would have option to use
# a the system available version

# Meta add the folder
set(META_MRAY_CMAKE_PREFIX_PATH ${MRAY_CONFIG_LIB_DIRECTORY}/cmake)
set(META_MRAY_PKGCONFIG_PREFIX_PATH ${MRAY_CONFIG_LIB_DIRECTORY}/pkgconfig)
append_cmake_prefix_path(${META_MRAY_CMAKE_PREFIX_PATH})
append_cmake_prefix_path(${META_MRAY_PKGCONFIG_PREFIX_PATH})
append_cmake_prefix_path(${MRAY_LIB_DIRECTORY})

# Get FMT
mray_build_ext_dependency_git(
        NAME fmt_ext
        URL "https://github.com/fmtlib/fmt.git"
        TAG "e57ca2e3685b160617d3d95fcd9e789c4e06ca88" # v10.1.0
        LICENSE_NAME "LICENSE.rst"
        BUILD_ARGS
            -DFMT_DEBUG_POSTFIX:STRING=
            -DFMT_DOC=OFF
            -DFMT_TEST=OFF
            #
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            # FMT has specific install variables for various stuff
            # set them specifically
            -DFMT_INC_DIR=Include
            -DFMT_USE_NONTYPE_TEMPLATE_ARGS=OFF
            -DFMT_LIB_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
            -DFMT_CMAKE_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>/cmake/fmt
            -DFMT_PKGCONFIG_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>/pkgconfig
)

# Get nlohmann-json
mray_build_ext_dependency_git(
        NAME nlohmann_json_ext
        URL "https://github.com/nlohmann/json.git"
        TAG "9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03" # v3.11.3
        LICENSE_NAME "LICENSE.MIT"
        BUILD_ARGS
            -DJSON_BuildTests=OFF
            -DBUILD_TESTING=OFF
            -DJSON_BuildTests=OFF
            -DJSON_MultipleHeaders=OFF
            -DNLOHMANN_JSON_CONFIG_INSTALL_DIR=${MRAY_CONFIG_LIB_DIRECTORY}/cmake/json
            -DNLOHMANN_JSON_INCLUDE_INSTALL_DIR=${MRAY_LIB_INCLUDE_DIRECTORY}
            -DNLOHMANN_JSON_PKGCONFIG_INSTALL_DIR=${MRAY_CONFIG_LIB_DIRECTORY}/pkgconfig
            # This is not cached and it install to a wrong folder :(
            -DNLOHMANN_ADD_NATVIS=OFF
)

# BS Thread pool
mray_build_ext_dependency_git(
        NAME thread_pool_ext
        URL "https://github.com/yalcinerbora/thread-pool.git"
        TAG "6c4ff304d09e0cdbfc085991d7c23fdfc0c48c1c" # v4.0.1 (Forked from original repo)
        LICENSE_NAME "LICENSE.txt"
        BUILD_ARGS
            -DTHREADPOOL_INSTALL=ON
)

# Get assimp
mray_build_ext_dependency_git(
    NAME assimp_ext
    URL "https://github.com/assimp/assimp.git"
    TAG "6a08c39e3a91ef385e76515cfad86aca4bfd57ff" # v.5.3.1
    LICENSE_NAME "LICENSE"
    BUILD_ARGS
        -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=ON
        -DASSIMP_BUILD_TESTS=OFF
        -DASSIMP_BUILD_ZLIB=ON
        -DASSIMP_INJECT_DEBUG_POSTFIX=OFF
        -DASSIMP_INSTALL=ON
        -DBUILD_SHARED_LIBS=ON
        -DASSIMP_HUNTER_ENABLED=OFF
        # Non-unix related
        -DASSIMP_LIB_INSTALL_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
        -DASSIMP_INCLUDE_INSTALL_DIR=Include
        -DASSIMP_BIN_INSTALL_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
        # Windows related
        -DLIBRARY_SUFFIX:STRING=
        -DUSE_STATIC_CRT=OFF
        # There is a bug on this one maybe?
        -DASSIMP_INSTALL_PDB=OFF
)

# Get GFG
mray_build_ext_dependency_git(
    NAME gfg_ext
    URL "https://github.com/yalcinerbora/GFGFileFormat.git"
    TAG "7634ebea392c8c3bb819b5f3a8ebc8ce2f3f63c8" # 30.03.2024 Latest Commit
    LICENSE_NAME "LICENSE"
    SOURCE_SUBDIR "Build/CMake"
    BUILD_ARGS
        -DGFG_INSTALL=ON
        -DGFG_BUILD_MAYA_PLUG_IN=OFF
)

# Get spdlog
mray_build_ext_dependency_git(
    NAME spdlog_ext
    URL "https://github.com/gabime/spdlog.git"
    TAG "ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36" # v.1.11.0
    LICENSE_NAME "LICENSE"
    BUILD_ARGS
        -DSPDLOG_BUILD_EXAMPLE=OFF
        -DSPDLOG_BUILD_PIC=ON
        -DSPDLOG_BUILD_SHARED=ON
        # fmt release and spdlog release does not match
        # due to iterator debug level (MSVC)
        -DSPDLOG_FMT_EXTERNAL=OFF
        -DCMAKE_DEBUG_POSTFIX=
)

# Get glbinding
# mray_build_ext_dependency_git(
#     NAME glbinding_ext
#     URL "https://github.com/yalcinerbora/glbinding.git"
#     TAG "10be400cdc5eba35c39a06cd6ca51f2b76d496ea" # Custom slightly modified version
#     LICENSE_NAME "LICENSE"
#     BUILD_ARGS
#         -DBUILD_SHARED_LIBS=ON
#         -DOPTION_BUILD_EXAMPLES=OFF
#         -DOPTION_BUILD_OWN_KHR_HEADERS=ON
#         -DOPTION_BUILD_TOOLS=OFF
#         # These are the added options
#         -DOPTION_SKIP_GLBINDING_AUX=ON
#         -DOPTION_ONLY_INSTALL_LARGE_HEADER=ON
#         # I am speculating this will be specifically usefull
#         # since the functions are actual gl functions instead of func. ptrs
#         -DOPTION_BUILD_WITH_LTO=ON
# )

# Get glfw
# mray_build_ext_dependency_git(
#     NAME glfw_ext
#     URL "https://github.com/glfw/glfw.git"
#     TAG "3fa2360720eeba1964df3c0ecf4b5df8648a8e52" # 17.06.2023 Latest Commit
#     LICENSE_NAME "LICENSE.md"
#     BUILD_ARGS
#         -DBUILD_SHARED_LIBS=ON
#         -DGLFW_BUILD_DOCS=OFF
#         -DGLFW_BUILD_EXAMPLES=OFF
#         -DGLFW_BUILD_TESTS=OFF
#         -DGLFW_INSTALL=ON
#         -DGLFW_USE_HYBRID_HPG=OFF
#         # Windows Related
#         -DGLFW_BUILD_WIN32=ON
#         -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON
# )

# Get CLI11
mray_build_ext_dependency_git(
    NAME cli11_ext
    URL "https://github.com/CLIUtils/CLI11.git"
    TAG "291c58789c031208f08f4f261a858b5b7083e8e2" # v2.3.2
    LICENSE_NAME "LICENSE"
    BUILD_ARGS
        -DBUILD_TESTING=OFF
        -DCLI11_BUILD_EXAMPLES=OFF
        -DCLI11_BUILD_EXAMPLES_JSON=OFF
        -DCLI11_BUILD_TESTS=OFF
        -DCLI11_SINGLE_FILE=OFF
)

# Get ImGUI
#include(DownloadDearImGUI.cmake)
# Get ImGUI Tex Inspect
#include(DownloadImGUITexInspect.cmake)

# Get gtest
if(MRAY_BUILD_TESTS)
    mray_build_ext_dependency_git(
        NAME googletest_ext
        URL "https://github.com/google/googletest.git"
        TAG "b796f7d44681514f58a683a3a71ff17c94edb0c1" # v1.13.0
        LICENSE_NAME "LICENSE"
        BUILD_ARGS
            -DBUILD_GMOCK=ON
            -DBUILD_SHARED_LIBS=ON
            # There is a bug on google test
            # it adds debug_prefix_NOT_FOUND when debug prefix is set
            # forgot to paste the link and i could not find it anymore
            -DCMAKE_DEBUG_POSTFIX=
    )
endif()

# Image Libraries compiled into oiio
add_subdirectory(oiio)

set(MRAY_ALL_EXT_DEP_TARGETS ${MRAY_ALL_EXT_DEP_TARGETS}
    ${MRAY_OIIO_EXT_DEP_TARGETS})

