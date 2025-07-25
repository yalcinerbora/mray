
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
            -DTBB_DIR:PATH=${MRAY_CONFIG_LIB_DIRECTORY}/cmake/TBB
        DEPENDENCIES
            tbb_ext
)