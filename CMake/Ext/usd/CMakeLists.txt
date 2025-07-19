
# TODO: If Linux, use the system library instead of building it from source
# (We require it already due to std::execution)
# TBB
mray_build_ext_dependency_git(
        NAME usd_tbb_ext
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
# And finally usd!
mray_build_ext_dependency_git(
        NAME usd_ext
        URL "https://github.com/PixarAnimationStudios/OpenUSD.git"
        TAG "1595c62ea8381b5b22eb8621afc8652f89b6136d" # v25.05.01

        # OpenUSD has it own structure and patching it will be
        # tedius. So it will have its own structure
        OVERRIDE_INSTALL_PREFIX ${MRAY_PLATFORM_LIB_DIRECTORY}/usd

        APPLY_PATCH "file://${MRAY_CMAKE_DIRECTORY}/Ext/usd/usd_fix_pair_template_instantiation.patch"

        LICENSE_NAME "LICENSE.txt"
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DPXR_BUILD_MONOLITHIC=OFF
            #
            -DPXR_BUILD_ANIMIX_TESTS=OFF
            -DPXR_BUILD_DOCUMENTATION=OFF
            -DPXR_BUILD_DRACO_PLUGIN=OFF
            -DPXR_BUILD_EMBREE_PLUGIN=OFF
            -DPXR_BUILD_EXAMPLES=OFF
            -DPXR_BUILD_HTML_DOCUMENTATION=OFF
            -DPXR_BUILD_IMAGING=OFF
            -DPXR_BUILD_MAYAPY_TESTS=OFF
            -DPXR_BUILD_OPENCOLORIO_PLUGIN=OFF
            -DPXR_BUILD_OPENIMAGEIO_PLUGIN=OFF
            -DPXR_BUILD_PRMAN_PLUGIN=OFF
            -DPXR_BUILD_PYTHON_DOCUMENTATION=OFF
            -DPXR_BUILD_TESTS=OFF
            -DPXR_BUILD_TUTORIALS=OFF
            -DPXR_BUILD_USDVIEW=OFF
            -DPXR_BUILD_USD_IMAGING=OFF
            -DPXR_BUILD_USD_TOOLS=OFF
            -DPXR_BUILD_USD_VALIDATION=OFF
            -DPXR_ENABLE_GL_SUPPORT=OFF
            -DPXR_ENABLE_HDF5_SUPPORT=OFF
            -DPXR_ENABLE_MATERIALX_SUPPORT=OFF
            -DPXR_ENABLE_METAL_SUPPORT=OFF
            -DPXR_ENABLE_NAMESPACES=ON
            -DPXR_ENABLE_OSL_SUPPORT=OFF
            # Bug on USD? Targets could not be located errors
            # "_pxr_enable_precompiled_header" on WSL
            -DPXR_ENABLE_PRECOMPILED_HEADERS=OFF
            -DPXR_ENABLE_PTEX_SUPPORT=OFF
            -DPXR_ENABLE_PYTHON_SUPPORT=OFF
            -DPXR_ENABLE_VULKAN_SUPPORT=OFF
            -DPXR_USE_DEBUG_PYTHON=OFF
            -DPXR_VALIDATE_GENERATED_CODE=OFF

            # TODO: Alembic is important
            # We need to enable it later (for simulation i/o)
            -DPXR_BUILD_ALEMBIC_PLUGIN=OFF
            # Similarly, OpenVDB (for volumetric data)
            -DPXR_ENABLE_OPENVDB_SUPPORT=OFF

            # TBB-related
            -DTBB_INCLUDE_DIRS=${MRAY_LIB_DIRECTORY}/Include
            -DTBB_LIBRARY=${MRAY_CONFIG_LIB_DIRECTORY}
            # This was required for TBB on windows
            -DTBB_USE_DEBUG_BUILD=$<IF:$<CONFIG:Debug>,ON,OFF>

        DEPENDENCIES
            usd_tbb_ext
)