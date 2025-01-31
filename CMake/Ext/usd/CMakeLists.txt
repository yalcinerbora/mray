
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
)

# OpenSubdiv
mray_build_ext_dependency_git(
    NAME usd_opensubdiv_ext
        URL "https://github.com/PixarAnimationStudios/OpenSubdiv"
        TAG "7d0ab5530feef693ac0a920585b5c663b80773b3" # v3.6.0
        LICENSE_NAME "LICENSE.txt"

        BUILD_ARGS
            -DCMAKE_LIBDIR_BASE=${MRAY_PLATFORM_NAME}/$<CONFIG>

            # Disable stuff
            -DNO_CLEW=ON
            -DNO_DOC=ON
            -DNO_EXAMPLES=ON
            -DNO_GLFW=ON
            -DNO_GLFW_X11=ON
            -DNO_GLTESTS=ON
            -DNO_MACOS_FRAMEWORK=ON
            -DNO_METAL=ON
            -DNO_OMP=ON
            -DNO_OPENCL=ON
            -DNO_OPENGL=ON
            -DNO_PTEX=ON
            -DNO_REGRESSION=ON
            -DNO_TBB=OFF
            -DNO_TESTS=ON
            -DNO_TUTORIALS=ON

            -DTBB_DIR=${META_MRAY_CMAKE_PREFIX_PATH}/TBB

            # TODO: After we support multiple backends,
            # we need to enable/disable selectively
            -DNO_CUDA=OFF
            -DNO_DX=ON
            -DNO_GLEW=ON

            # Disable static runtime?
            -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF

        DEPENDENCIES
            usd_tbb_ext
)

# And finally usd!
set(USD_TBB_LIB_DEBUG ${MRAY_CONFIG_LIB_DIRECTORY}/tbb_debug${CMAKE_LINK_LIBRARY_SUFFIX})
set(USD_TBB_LIB_RELEASE ${MRAY_CONFIG_LIB_DIRECTORY}/tbb${CMAKE_LINK_LIBRARY_SUFFIX})
set(USD_OSD_LIB_CPU ${MRAY_CONFIG_LIB_DIRECTORY}/osdCPU$<$<CONFIG:Debug>:d>${CMAKE_LINK_LIBRARY_SUFFIX})
set(USD_OSD_LIB_GPU ${MRAY_CONFIG_LIB_DIRECTORY}/osdGPU$<$<CONFIG:Debug>:d>${CMAKE_LINK_LIBRARY_SUFFIX})

mray_build_ext_dependency_git(
        NAME usd_ext
        URL "https://github.com/PixarAnimationStudios/OpenUSD.git"
        TAG "9b0c13b2efa6233c8a4a4af411833628c5435bde" # v24.11

        # OpenUSD has it own structure and patching it will be
        # tedius. So it will have its own structure
        OVERRIDE_INSTALL_PREFIX ${MRAY_PLATFORM_LIB_DIRECTORY}/usd

        LICENSE_NAME "LICENSE.txt"
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=ON
            -DPXR_BUILD_MONOLITHIC=ON

            #-DTBB_ROOT_DIR=${META_MRAY_CMAKE_PREFIX_PATH}
            -DTBB_INCLUDE_DIRS=${MRAY_LIB_DIRECTORY}/Include
            -DTBB_LIBRARY=${MRAY_CONFIG_LIB_DIRECTORY}
            # -DTBB_tbb_LIBRARY_DEBUG=${TBB_LIB_DEBUG}
            # -DTBB_tbb_LIBRARY_RELEASE=${TBB_LIB_RELEASE}
            # This is required for TBB on windows
            -DTBB_USE_DEBUG_BUILD=$<IF:$<CONFIG:Debug>,ON,OFF>

            #-DOPENSUBDIV_ROOT_DIR=${META_MRAY_CMAKE_PREFIX_PATH}/OpenSubdiv
            -DOPENSUBDIV_INCLUDE_DIR=${MRAY_LIB_DIRECTORY}/Include
            -DOPENSUBDIV_OSDCPU_LIBRARY=${USD_OSD_LIB_CPU}
            -DOPENSUBDIV_OSDGPU_LIBRARY=${USD_OSD_LIB_GPU}

            -DPXR_BUILD_ANIMIX_TESTS=OFF
            -DPXR_BUILD_DOCUMENTATION=OFF
            -DPXR_BUILD_DRACO_PLUGIN=OFF
            -DPXR_BUILD_EMBREE_PLUGIN=OFF
            -DPXR_BUILD_EXAMPLES=OFF
            -DPXR_BUILD_HTML_DOCUMENTATION=OFF
            -DPXR_BUILD_MAYAPY_TESTS=OFF
            -DPXR_BUILD_OPENCOLORIO_PLUGIN=OFF
            -DPXR_BUILD_OPENIMAGEIO_PLUGIN=OFF
            -DPXR_BUILD_PRMAN_PLUGIN=OFF
            -DPXR_BUILD_PYTHON_DOCUMENTATION=OFF
            -DPXR_BUILD_TESTS=OFF
            -DPXR_BUILD_TUTORIALS=OFF
            -DPXR_BUILD_USDVIEW=OFF
            -DPXR_BUILD_USD_TOOLS=OFF
            -DPXR_BUILD_USD_VALIDATION=OFF
            -DPXR_ENABLE_GL_SUPPORT=OFF
            -DPXR_ENABLE_HDF5_SUPPORT=OFF
            -DPXR_ENABLE_MATERIALX_SUPPORT=OFF
            -DPXR_ENABLE_METAL_SUPPORT=OFF
            -DPXR_ENABLE_NAMESPACES=ON
            -DPXR_ENABLE_OSL_SUPPORT=OFF
            -DPXR_ENABLE_PRECOMPILED_HEADERS=ON
            -DPXR_ENABLE_PTEX_SUPPORT=OFF
            -DPXR_ENABLE_PYTHON_SUPPORT=OFF
            -DPXR_ENABLE_VULKAN_SUPPORT=OFF
            -DPXR_USE_BOOST_PYTHON=OFF
            -DPXR_USE_DEBUG_PYTHON=OFF
            -DPXR_VALIDATE_GENERATED_CODE=OFF

            -DPXR_BUILD_USD_IMAGING=ON
            -DPXR_BUILD_IMAGING=ON

            # TODO: Alembic is important
            # We need to enable it later (for simulation i/o)
            -DPXR_BUILD_ALEMBIC_PLUGIN=OFF
            # Similarly, OpenVDB (for volumetric data)
            -DPXR_ENABLE_OPENVDB_SUPPORT=OFF

        DEPENDENCIES
            usd_tbb_ext
            usd_opensubdiv_ext
)