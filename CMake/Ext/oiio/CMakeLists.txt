# For OIIO, we embed everything to its dll so we do not need these files to
# be present after the mega lib generation. So override install directory
# of these modules (even do not do install at all)

# And finally call the actual oiio external project add

# Install only OIIO related libraries to a different folder
# this folder will be deleted after everything is installed
# Thus, all of these libraries should be static library
set(OIIO_TMP_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/tmp)
set(OIIO_LIB_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/tmp/lib)

# Append to prefix path here for lib finding
set(OIIO_LIB_CONFIG_DIR ${OIIO_LIB_DIR}/${MRAY_PLATFORM_NAME}/$<CONFIG>)
append_cmake_prefix_path(${OIIO_LIB_CONFIG_DIR}/cmake)
append_cmake_prefix_path(${OIIO_LIB_CONFIG_DIR}/pkgconfig)

# Normally MRay uses structure of arrays style library directory structure
# which is easy for the consumer libraries to find the actual lib/dlls
#
# i.e.
# Lib/Include/<LIB> (Shared between configs)
# Lib/Platform/Configuration/<LIB>.dll
# Lib/Platform/Configuration/cmake/<LIB>Config.cmake
# etc..
#
# However i could not get it to work for the oiio :(
# This hierarchy uses AoS style of hierarchy.
# these should almost always be static library
# Ext/oiio/Lib/Platform/Configuration/<LIB>/....
# This will duplicate the includes :(
# Thank god boost is found using the SoA method
# Libjpeg turbo could not be found so using that method for it.

# ZLIB
set(ZLIB_INSTALL_LOC ${OIIO_TMP_DIR}/zlib)
mray_build_ext_dependency_git(
        NAME oiio_zlib_ext
        URL "https://github.com/madler/zlib.git"
        TAG "04f42ceca40f73e2978b50e93806c2a18c1281fc" # v1.2.13
        LICENSE_NAME "LICENSE"

        OVERRIDE_INSTALL_PREFIX ${ZLIB_INSTALL_LOC}
        BUILD_ARGS
            # Directly write in a folder
            # Zlib stores these variables between configs
            # wich is bad. When you switch to release it still installs
            # the debug config. So we need to write these
            -DINSTALL_BIN_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_INC_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_LIB_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_MAN_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_PKGCONFIG_DIR=${ZLIB_INSTALL_LOC}
            -DCMAKE_DEBUG_POSTFIX:STRING=d

)
append_cmake_prefix_path(${ZLIB_INSTALL_LOC})

# Copy zlib.dll/so to "Lib"
if(UNIX AND NOT APPLE)
    set(MRAY_ZLIB_COPY_NAME libz$<$<CONFIG:Debug>:d>${CMAKE_SHARED_LIBRARY_SUFFIX}.1)
elseif(MSVC)
    set(MRAY_ZLIB_COPY_NAME zlib$<$<CONFIG:Debug>:d>${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()
ExternalProject_Add_Step(oiio_zlib_ext install_extra
                WORKING_DIRECTORY ${ZLIB_INSTALL_LOC}/
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${MRAY_ZLIB_COPY_NAME}
                        ${MRAY_CONFIG_LIB_DIRECTORY}
                DEPENDEES install
                COMMENT "Copy zlib to Lib folder")

# OpenEXR & IMath
set(IMATH_INSTALL_LOC ${OIIO_TMP_DIR}/imath)
mray_build_ext_dependency_git(
        NAME imath_ext
        URL "https://github.com/AcademySoftwareFoundation/Imath.git"
        TAG "c0396a055a01bc537d32f435aee11a9b7ed6f0b5" # v3.1.12

        LICENSE_NAME "LICENSE.md"
        # OpenImageIO uses Imath as a subdirectory on includes.
        # Although we do not use it, we need to install it to proper folder
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${IMATH_INSTALL_LOC}

        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_TESTING=OFF
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            -DPYTHON=OFF
            -DDOCS=OFF
            -DIMATH_HALF_USE_LOOKUP_TABLE=OFF
)
append_cmake_prefix_path(${IMATH_INSTALL_LOC})

set(OPENEXR_INSTALL_LOC ${OIIO_TMP_DIR}/openexr)
mray_build_ext_dependency_git(
        NAME openexr_ext
        URL "https://github.com/AcademySoftwareFoundation/openexr.git"
        TAG "7ec1cf59a0409d9d5c60ed111da9831890fa8296" # v3.3.4

        LICENSE_NAME "LICENSE.md"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${OPENEXR_INSTALL_LOC}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DOPENEXR_BUILD_TOOLS=OFF
            -DOPENEXR_FORCE_INTERNAL_IMATH=OFF
            -DOPENEXR_INSTALL=ON
            -DOPENEXR_INSTALL_EXAMPLES=OFF
            -DOPENEXR_INSTALL_PKG_CONFIG=OFF
            -DOPENEXR_INSTALL_TOOLS=OFF
            -DBUILD_TESTING=OFF
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            #
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
            -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
            #
            -DImath_DIR:PATH=${IMATH_INSTALL_LOC}/lib/cmake/Imath

        DEPENDENCIES
            imath_ext
)
append_cmake_prefix_path(${OPENEXR_INSTALL_LOC})

# Lib TIFF
set(LIBTIFF_INSTALL_LOC ${OIIO_TMP_DIR}/libtiff)
mray_build_ext_dependency_git(
        NAME libtiff_ext
        URL "https://gitlab.com/libtiff/libtiff.git"
        TAG "9dff73bebc5661f2dace6f16e14cf9e857172f4e" # v4.7.0

        APPLY_PATCH "file://${MRAY_CMAKE_DIRECTORY}/Ext/oiio/libtiff_remove_cmath.patch"

        LICENSE_NAME "LICENSE.md"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${LIBTIFF_INSTALL_LOC}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -Dtiff-tools=OFF
            -Dtiff-tests=OFF
            -Dtiff-docs=OFF
            -Dtiff-contrib=OFF
            -Dtiff-install=ON
            -Dzlib=ON
            -DZLIB_ROOT:PATH=${ZLIB_INSTALL_LOC}
            -DJPEG_ROOT:PATH=${JPEGTURBO_INSTALL_LOC}
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            # Finds stuff from miniconda since it is on path
            # It may be on path like in my system
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
            -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
            #
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        DEPENDENCIES
            oiio_zlib_ext
            libjpeg_turbo_ext
            libpng_ext
)
append_cmake_prefix_path(${LIBTIFF_INSTALL_LOC})

# Lib JPEG-turbo
set(JPEGTURBO_INSTALL_LOC ${OIIO_TMP_DIR}/jpegTurbo)
mray_build_ext_dependency_git(
        NAME libjpeg_turbo_ext
        URL "https://github.com/libjpeg-turbo/libjpeg-turbo.git"
        TAG "6c87537f60941f3c265c339fe60d1e31d2a42ccf" # v3.0.0.0

        LICENSE_NAME "LICENSE.md"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${JPEGTURBO_INSTALL_LOC}
        BUILD_ARGS
            -DENABLE_SHARED=OFF
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)
append_cmake_prefix_path(${JPEGTURBO_INSTALL_LOC})

# Lib PNG
set(LIBPNG_INSTALL_LOC ${OIIO_TMP_DIR}/libPNG)
mray_build_ext_dependency_git(
        NAME libpng_ext
        URL "https://github.com/pnggroup/libpng.git"
        TAG "ea127968204cc5d10f3fc9250c306b9e8cbd9b80" # v1.6.48

        LICENSE_NAME "LICENSE"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${LIBPNG_INSTALL_LOC}
        BUILD_ARGS
            -DPNG_BUILD_ZLIB=OFF
            -DPNG_EXECUTABLES=OFF
            -DPNG_TOOLS=OFF
            -DPNG_HARDWARE_OPTIMIZATIONS=ON
            -DPNG_FRAMEWORK=OFF
            -DPNG_SHARED=OFF
            -DPNG_STATIC=ON
            -DPNG_TESTS=OFF
            -DPNG_DEBUG_POSTFIX:STRING=d
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON

        DEPENDENCIES
            oiio_zlib_ext
)
append_cmake_prefix_path(${LIBPNG_INSTALL_LOC})

# OpenColorIO and Sidekicks
set(MINIZIP_NG_INSTALL_LOC ${OIIO_TMP_DIR}/minizip_ng)
mray_build_ext_dependency_git(
        NAME minizip_ng_ext
        URL "https://github.com/zlib-ng/minizip-ng.git"
        TAG "241428886216f0f0efd6926efcaaaa13794e51bd" # v3.0.7

        LICENSE_NAME "LICENSE"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${MINIZIP_NG_INSTALL_LOC}
        BUILD_ARGS

            -DBUILD_SHARED_LIBS=OFF
            -DMZ_OPENSSL=OFF
            -DMZ_LIBBSD=OFF
            -DMZ_BUILD_TESTS=OFF
            -DMZ_COMPAT=OFF
            -DMZ_BZIP2=OFF
            -DMZ_LZMA=OFF
            -DMZ_LIBCOMP=OFF
            -DMZ_ZSTD=OFF
            -DMZ_PKCRYPT=OFF
            -DMZ_WZAES=OFF
            -DMZ_SIGNING=OFF
            -DMZ_ZLIB=ON
            -DMZ_ICONV=OFF
            -DMZ_FETCH_LIBS=OFF
            -DMZ_FORCE_FETCH_LIBS=OFF
            -DZLIB_LIBRARY=${ZLIB_LIBRARIES}
            -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS}
        DEPENDENCIES
            oiio_zlib_ext

)

set(EXPAT_INSTALL_LOC ${OIIO_TMP_DIR}/expat)
mray_build_ext_dependency_git(
        NAME expat_ext
        URL "https://github.com/libexpat/libexpat.git"
        TAG "654d2de0da85662fcc7644a7acd7c2dd2cfb21f0" # v2.5.0

        LICENSE_NAME "COPYING"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${EXPAT_INSTALL_LOC}
        SOURCE_SUBDIR "expat"
        BUILD_ARGS

            -DEXPAT_BUILD_DOCS=OFF
            -DEXPAT_BUILD_EXAMPLES=OFF
            -DEXPAT_BUILD_TESTS=OFF
            -DEXPAT_BUILD_TOOLS=OFF
            -DEXPAT_SHARED_LIBS=OFF

)

set(YAML_CPP_INSTALL_LOC ${OIIO_TMP_DIR}/yaml_cpp)
mray_build_ext_dependency_git(
        NAME yaml_cpp_ext
        URL "https://github.com/jbeder/yaml-cpp.git"
        TAG "f7320141120f720aecc4c32be25586e7da9eb978" # v0.8.0

        LICENSE_NAME "LICENSE"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${YAML_CPP_INSTALL_LOC}
        BUILD_ARGS

            -DBUILD_SHARED_LIBS=OFF
            -DYAML_BUILD_SHARED_LIBS=OFF
            -DYAML_CPP_BUILD_TESTS=OFF
            -DYAML_CPP_BUILD_TOOLS=OFF
            -DYAML_CPP_BUILD_CONTRIB=OFF

)

# Inject an custom cmakelists.txt and make cmake to use it
set(PYSTRING_INSTALL_LOC ${OIIO_TMP_DIR}/pystring)
mray_build_ext_dependency_git(
        NAME pystring_ext
        URL "https://github.com/imageworks/pystring.git"
        TAG "c2de99deb4f0bd13751f8436400b5e8662301769" # v1.1.3

        LICENSE_NAME "LICENSE"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${PYSTRING_INSTALL_LOC}
)
ExternalProject_Get_property(pystring_ext SOURCE_DIR)
set(MRAY_PYSTRING_SRC_LOCATION ${SOURCE_DIR})
# Inject an custom CMakeLists.txt and make cmake to use it
ExternalProject_Add_Step(pystring_ext inject_cmake
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists-pystring.cmake
                        ${MRAY_PYSTRING_SRC_LOCATION}/CMakeLists.txt
                DEPENDEES download update patch
                DEPENDERS configure
                COMMENT "Injecting a cmake lists to pystring"
)

set(OCIO_INSTALL_LOC ${OIIO_TMP_DIR}/OpenColorIO)
mray_build_ext_dependency_git(
        NAME ocio_ext
        URL "https://github.com/AcademySoftwareFoundation/OpenColorIO.git"
        TAG "6918fad3f5d22ac3ef2397c754bf4268c2b58dd0" # v2.4.2

        APPLY_PATCH "file://${MRAY_CMAKE_DIRECTORY}/Ext/oiio/occio_add_config_to_pystring.patch"

        LICENSE_NAME "LICENSE"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${OCIO_INSTALL_LOC}
        BUILD_ARGS
            # TODO: Change this later(oiio has "print" clashes with fmt)
            -DCMAKE_CXX_STANDARD=20
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF

            -DOCIO_BUILD_APPS=OFF
            -DOCIO_BUILD_DOCS=OFF
            -DOCIO_BUILD_FROZEN_DOCS=OFF
            -DOCIO_BUILD_GPU_TESTS=OFF
            -DOCIO_BUILD_JAVA=OFF
            -DOCIO_BUILD_NUKE=OFF
            -DOCIO_BUILD_PYTHON=OFF
            -DOCIO_BUILD_TESTS=OFF
            -DOCIO_INSTALL_EXT_PACKAGES=NONE

            # ZLIB does not have config
            -DZLIB_ROOT:PATH=${ZLIB_INSTALL_LOC}
            # We create config for pystring
            -Dpystring_DIR:PATH=${PYSTRING_INSTALL_LOC}/lib/cmake/pystring
            # For the rest, configs should be fine
            -DImath_DIR:PATH=${IMATH_INSTALL_LOC}/lib/cmake/Imath
            -Dminizip-ng_DIR:PATH=${MINIZIP_NG_INSTALL_LOC}/lib/cmake/minizip-ng
            -Dexpat_DIR:PATH=${EXPAT_INSTALL_LOC}/lib/cmake/expat-2.5.0
            -Dyaml-cpp_DIR:PATH=${YAML_CPP_INSTALL_LOC}/lib/cmake/yaml-cpp

        DEPENDENCIES
            imath_ext
            oiio_zlib_ext
            expat_ext
            minizip_ng_ext
            yaml_cpp_ext
            pystring_ext
)
append_cmake_prefix_path(${OCIO_INSTALL_LOC})

# TODO: Add Jpeg2000, HEIF, HDR

# MSVC requires do static linking, but on linux it must be off
# due to linking with system libraries....
set(DO_STATIC_LINKING OFF)
if(WIN32)
    set(DO_STATIC_LINKING ON)
endif()

# And finally oiio
mray_build_ext_dependency_git(
        NAME oiio_ext
        URL "https://github.com/OpenImageIO/oiio.git"
        TAG "e24cac1b35f0e0bec7f884571e4bfad104ec5e73" # v3.0.8.1
        LICENSE_NAME "LICENSE.md"

        APPLY_PATCH "file://${MRAY_CMAKE_DIRECTORY}/Ext/oiio/oiio_change_old_png_include.patch"

        BUILD_ARGS
            # OIIO caches xxx_ROOT variables
            # and libaries does not change between configs
            --fresh
            # TODO: Change this later(oiio has "print" clashes with fmt)
            -DCMAKE_CXX_STANDARD=20
            -DEMBEDPLUGINS=ON
            -DUSE_PYTHON=OFF
            -DUSE_QT=OFF

            -DBUILD_SHARED_LIBS=ON
            -DLINKSTATIC=${DO_STATIC_LINKING}
            -DOIIO_BUILD_TESTS=OFF
            -DOIIO_BUILD_TOOLS=OFF
            -DOIIO_INTERNALIZE_FMT=OFF # On Linux this was required
            -DBUILD_DOCS=OFF
            -DBUILD_TESTING=OFF
            -DINSTALL_DOCS=OFF
            -DBUILD_MISSING_FMT=OFF # We have a fmt
            -DOpenImageIO_ALWAYS_PREFER_CONFIG=ON
            -DSTOP_ON_WARNING=OFF
            -DVERBOSE=OFF
            # Do not use system (clash with miniconda etc...)
            -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
            -DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=OFF
            -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
            # Do not install msvc runtime libraries
            -DCMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP=ON
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}

            # ZLIB do not have cmake config
            -DZLIB_ROOT:PATH=${ZLIB_INSTALL_LOC}
            # For the rest, configs should be fine
            -DImath_DIR:PATH=${IMATH_INSTALL_LOC}/lib/cmake/Imath
            -DOpenEXR_DIR:PATH=${OPENEXR_INSTALL_LOC}/lib/cmake/OpenEXR
            -DJPEGTurbo_DIR:PATH=${JPEGTURBO_INSTALL_LOC}/lib/cmake/libjpeg-turbo
            -DJPEG_DIR:PATH=${JPEGTURBO_INSTALL_LOC}/lib/cmake/libjpeg-turbo
            -DTIFF_DIR:PATH=${LIBTIFF_INSTALL_LOC}/lib/cmake/tiff
            -DPNG_DIR:PATH=${PNG_INSTALL_LOC}/lib/cmake/PNG
            #
            -DOpenColorIO_DIR:PATH=${PNG_INSTALL_LOC}/lib/cmake/OpenColorIO
            -Dpystring_DIR:PATH=${PYSTRING_INSTALL_LOC}/lib/cmake/pystring
            -DImath_DIR:PATH=${IMATH_INSTALL_LOC}/lib/cmake/Imath
            -Dminizip-ng_DIR:PATH=${MINIZIP_NG_INSTALL_LOC}/lib/cmake/minizip-ng
            -Dexpat_DIR:PATH=${EXPAT_INSTALL_LOC}/lib/cmake/expat-2.5.0
            -Dyaml-cpp_DIR:PATH=${YAML_CPP_INSTALL_LOC}/lib/cmake/yaml-cpp

        DEPENDENCIES
            openexr_ext
            libtiff_ext
            libjpeg_turbo_ext
            libpng_ext
            fmt_ext
            oiio_zlib_ext
            ocio_ext
)

# Get the name of the deps outside
set(MRAY_OIIO_EXT_DEP_TARGETS
    ${MRAY_ALL_EXT_DEP_TARGETS} PARENT_SCOPE)