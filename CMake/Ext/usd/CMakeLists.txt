set(USD_TMP_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/tmp)
set(USD_LIB_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/tmp/lib)

# Normally MRay uses structure of arrays style library directory structure.
# Similar difference like oiio, please check that comment

# TBB
#set(TBB_INSTALL_LOC ${USD_TMP_DIR}/tbb)
mray_build_ext_dependency_git(
    NAME usd_tbb_ext
        URL "https://github.com/uxlfoundation/oneTBB"
        TAG "0c0ff192a2304e114bc9e6557582dfba101360ff" # v2021.9.0
        LICENSE_NAME "LICENSE.txt"

        APPLY_PATCH "https://github.com/uxlfoundation/oneTBB/commit/18bd322d92e5820ad76d09614ea1a2770ef95a97.patch"

        #OVERRIDE_INSTALL_PREFIX ${TBB_INSTALL_LOC}
        BUILD_ARGS
            -DTBB_TEST=OFF
            -DTBB_STRICT=ON
)

# OpenSubdiv
#set(OPENSUBDIV_INSTALL_LOC ${USD_TMP_DIR}/opensubdiv)
mray_build_ext_dependency_git(
    NAME usd_opensubdiv_ext
        URL "https://github.com/PixarAnimationStudios/OpenSubdiv"
        TAG "7d0ab5530feef693ac0a920585b5c663b80773b3" # v3.6.0
        LICENSE_NAME "LICENSE.txt"

        #OVERRIDE_INSTALL_PREFIX ${OPENSUBDIV_INSTALL_LOC}
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

            -DTBB_DIR=${TBB_INSTALL_LOC}/lib/cmake/TBB

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

# # And finally usd!
# mray_build_ext_dependency_git(
#         NAME oiio_ext
#         URL "https://github.com/OpenImageIO/oiio.git"
#         TAG "feb28fd2647b088f5b083621a98fec97fddaf167" # Latest commit as  of  05.05.2024

#         LICENSE_NAME "LICENSE.md"
#         BUILD_ARGS
#           ......

#         DEPENDENCIES
#             usd_tbb_ext
#             usd_opensubdiv_ext
# )