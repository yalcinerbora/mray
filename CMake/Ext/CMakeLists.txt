cmake_minimum_required(VERSION 3.26)

# CMAKE Project Definitions and Source, Library Folders are separate
# Directories
set(MRAY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
cmake_path(NORMAL_PATH MRAY_DIRECTORY)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MRAY_DIRECTORY}CMake/Include")

# Do not use in source builds
include("PreventInSourceBuilds")

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Configuration types, for multi-config generators (MRay supports only Debug and Release for external libraries)")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Configuration type for single-config generators. (MRay supports only Debug and Release for external libraries)")

# ExternalProject_Add warpper
include(ExtDownloadDeps)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create a project..
project(MRayExternal DESCRIPTION "MRay dependency downloader/compiler")

option(MRAY_BUILD_TESTS "Build Test Libraries (gtest)" ON)
option(MRAY_BUILD_VISOR "Build Visor Libraries (glfw etc..)" ON)
option(MRAY_BUILD_USD "Build USD Libraries (usd/tbb/opensubdiv)" OFF)
option(MRAY_DL_EMBREE "Download Embree Libraries (embree4)" OFF)
option(MRAY_BUILD_TRACY "Build \"Tracy\" (open-source profiler)")

# Populate the meta directories for this project hierarchy
include("Macros")
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
    TAG "4ceb47e03347088e95ed98034c9509883f67e98a" # 12.06.2024 Latest Commit
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
    TAG "27cb4c76708608465c413f6d0e6b8d99a4d84302" # v.1.14.1
    LICENSE_NAME "LICENSE"
    BUILD_ARGS
        -DSPDLOG_BUILD_EXAMPLE=OFF
        -DSPDLOG_BUILD_PIC=ON
        -DSPDLOG_BUILD_SHARED=ON
        # fmt release and spdlog release does not match
        # due to iterator debug level (MSVC)
        -DSPDLOG_FMT_EXTERNAL=ON
        -DCMAKE_DEBUG_POSTFIX=
    DEPENDENCIES
        fmt_ext
)

# Get CLI11
mray_build_ext_dependency_git(
    NAME cli11_ext
    URL "https://github.com/CLIUtils/CLI11.git"
    TAG "4160d259d961cd393fd8d67590a8c7d210207348" # v2.5.0
    LICENSE_NAME "LICENSE"

    APPLY_PATCH "file://${MRAY_CMAKE_DIRECTORY}/Ext/cli11_fix_nested_footer_print.patch"

    BUILD_ARGS
        -DBUILD_TESTING=OFF
        -DCLI11_BUILD_EXAMPLES=OFF
        -DCLI11_BUILD_EXAMPLES_JSON=OFF
        -DCLI11_BUILD_TESTS=OFF
        -DCLI11_SINGLE_FILE=OFF
)

if(MRAY_BUILD_VISOR)

    # Get glfw
    mray_build_ext_dependency_git(
        NAME glfw_ext
        URL "https://github.com/glfw/glfw.git"
        TAG "3fa2360720eeba1964df3c0ecf4b5df8648a8e52" # 17.06.2023 Latest Commit
        LICENSE_NAME "LICENSE.md"
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=ON
            -DGLFW_BUILD_DOCS=OFF
            -DGLFW_BUILD_EXAMPLES=OFF
            -DGLFW_BUILD_TESTS=OFF
            -DGLFW_INSTALL=ON
            -DGLFW_USE_HYBRID_HPG=OFF
            # Windows Related
            -DGLFW_BUILD_WIN32=ON
            -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON
    )

    # Get vulkan-related
    mray_build_ext_dependency_git(
        NAME vulkan_headers_ext
        URL "https://github.com/KhronosGroup/Vulkan-Headers.git"
        TAG "577baa05033cf1d9236b3d078ca4b3269ed87a2b" # 1.3.280
        LICENSE_NAME "LICENSE.md"
        FORCE_RELEASE
        BUILD_ARGS
            -DUPDATE_DEPS=OFF
            -DBUILD_TESTS=OFF
            -DVULKAN_HEADERS_INSTALL=ON
    )

    mray_build_ext_dependency_git(
        NAME vulkan_loader_ext
        URL "https://github.com/KhronosGroup/Vulkan-Loader.git"
        TAG "61a9c50248e09f3a0e0be7ce6f8bb1663855f979" # 1.3.280
        LICENSE_NAME "LICENSE.txt"
        FORCE_RELEASE
        BUILD_ARGS
            -DUPDATE_DEPS=OFF
            -DBUILD_TESTS=OFF
            -DVULKAN_HEADERS_INSTALL_DIR=${MRAY_LIB_DIRECTORY}/share

        DEPENDENCIES
            vulkan_headers_ext
    )

    # Get ImGUI
    include(DownloadDearImGUI.cmake)
    # Get ImGUI Tex Inspect
    #include(DownloadImGUITexInspect.cmake)

    # slang shader compiler
    # this is a bit tricky, since we will not build it
    # but we will download the executable and copy it.
    # so external project should not work? (TODO: Check this)
    include(FetchContent)
    # More importantly, we are downloading an pre-compiled executable,
    # validate it with a hash at least.
    # Slang does not provide a sha256 hash directly in github, so we will create it
    # manually.
    # !!!!
    # THE HASHES ARE GENERATED ON LINUX SYSTEM "sha256sum v9.4" MANUALLY!
    # Technically you are still trusting me, please be advised.
    # !!!
    # Force TLS certificate check
    set(CMAKE_TLS_VERIFY ON)

    # Slang version v.2024.14.4
    if(WIN32)
        set(SLANGC_ZIP_NAME "slang-2025.8-windows-x86_64.zip")
        set(SLANGC_SHA256_HASH "21D4F2786BBE7E23B79F8186C2BEA6782BB3FF0CFB3BA26FB487DE1019EC08D7")
    elseif(UNIX AND NOT APPLE)
        set(SLANGC_ZIP_NAME "slang-2025.8-linux-x86_64.tar.gz")
        set(SLANGC_SHA256_HASH "E94ED4CC6372B9B028844B6874049F8DBB0AE2B5E174323D9B52BEF327C8B4BB")
    else()
        message(FATAL_ERROR "Unsupported platform for macos(?)")
    endif()

    FetchContent_Declare(slangc_ext
        URL "https://github.com/shader-slang/slang/releases/download/v2025.8/${SLANGC_ZIP_NAME}"
        URL_HASH "SHA256=${SLANGC_SHA256_HASH}")
    FetchContent_MakeAvailable(slangc_ext)

    # Now write it to the MRAY_LIB_DIRECTORY
    file(GLOB SLANGC_BINARIES
        "${slangc_ext_SOURCE_DIR}/bin/*"
        "${slangc_ext_SOURCE_DIR}/lib/*")
    file(COPY ${SLANGC_BINARIES} DESTINATION ${MRAY_PLATFORM_LIB_DIRECTORY}/slang)

endif()

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

if(MRAY_DL_EMBREE)
    # Embree v.4.4.0 with SYCL
    if(WIN32)
        set(EMBREE_ZIP_NAME "embree-4.4.0.x64.windows.zip")
        set(EMBREE_SHA256_HASH "D951E5E6BD295C54CDD66BE9CDB44A4E8C42FB38A99F94F79305E48765FC3454")
    elseif(UNIX AND NOT APPLE)
        set(EMBREE_ZIP_NAME "embree-4.4.0.x86_64.linux.tar.gz")
        set(EMBREE_SHA256_HASH "CB3D4402537FC9165C76C3316B8953DCFEA523CD1EAF588E2DE7639864EE3C57")
    else()
        message(FATAL_ERROR "Unsupported platform for macos(?)")
    endif()

    FetchContent_Declare(embree_ext
        URL "https://github.com/RenderKit/embree/releases/download/v4.4.0/${EMBREE_ZIP_NAME}"
        URL_HASH "SHA256=${EMBREE_SHA256_HASH}")
    FetchContent_MakeAvailable(embree_ext)

    # Now write it to the MRAY_LIB_DIRECTORY
    file(GLOB EMBREE_BINARIES
        "${embree_ext_SOURCE_DIR}/lib/*"
        "${embree_ext_SOURCE_DIR}/bin/*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    file(GLOB EMBREE_INCLUDES
        "${embree_ext_SOURCE_DIR}/include/*")
    file(COPY ${EMBREE_INCLUDES} DESTINATION ${MRAY_PLATFORM_LIB_DIRECTORY}/embree/include)
    file(COPY ${EMBREE_BINARIES} DESTINATION ${MRAY_PLATFORM_LIB_DIRECTORY}/embree/lib)

endif()

if(MRAY_DL_EMBREE OR MRAY_BUILD_USD)
    # TBB
    mray_build_ext_dependency_git(
        NAME tbb_ext
        URL "https://github.com/uxlfoundation/oneTBB"
        TAG "a00cc3b8b5fb4d8115e9de56bf713157073ed68c" # v2021.9.0
        LICENSE_NAME "LICENSE.txt"

        APPLY_PATCH "https://github.com/uxlfoundation/oneTBB/commit/a444e95134122ee75ccbdf0e6d3315ba458506f4.patch"

        BUILD_ARGS
            -DTBB_TEST=OFF
            -DTBB_STRICT=ON

            # CMake 4.0.0 onwards dropped support for 3.5,
            # most libs have this set as minimum. Try to force it
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )
endif()

if(MRAY_BUILD_TRACY)
    # Tracy
    mray_build_ext_dependency_git(
        NAME tracy_ext
        URL "https://github.com/wolfpld/tracy.git"
        TAG "c556831ddc6fe26d2fce01c14c97205a9dad46d5" # v0.12.2
        LICENSE_NAME "LICENSE"

        BUILD_ARGS
            -DTRACY_STATIC=OFF
            -DTRACY_ENABLE=ON
            -DTRACY_NO_VSYNC_CAPTURE=ON
            -DTRACY_NO_EXIT=OFF
            -DTRACY_NO_FRAME_IMAGE=ON
            -DTRACY_ON_DEMAND=ON
            -DTRACY_ONLY_LOCALHOST=ON
            -DTRACY_DELAYED_INIT=ON
            -DTRACY_MANUAL_LIFETIME=ON
    )
endif()

# The boss!
# Image Libraries compiled into oiio
add_subdirectory(oiio)

# The second boss! (Mush simpler than oiio though)
if(MRAY_BUILD_USD)
    add_subdirectory(usd)
endif()

set(MRAY_ALL_EXT_DEP_TARGETS ${MRAY_ALL_EXT_DEP_TARGETS}
    ${MRAY_OIIO_EXT_DEP_TARGETS})

