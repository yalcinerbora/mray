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

# This one is tedious. It requires boost :(
# Now external project add boost with proper flags and submodules
# Clone boost
# Come on boost, too many deps :(
set(OIIO_BOOST_SUBMODULE_LIST
    # CMake
    tools/cmake
    # FS Related
    libs/filesystem
    libs/assert libs/bind libs/function
    libs/config libs/container_hash libs/core
    libs/describe libs/mp11 libs/static_assert
    libs/type_traits libs/throw_exception libs/detail
    libs/preprocessor libs/io libs/iterator
    libs/smart_ptr libs/concept_check libs/move
    libs/conversion libs/system libs/predef
    libs/typeof libs/function_types libs/variant2
    libs/atomic libs/mpl libs/fusion
    libs/align libs/tuple libs/optional
    libs/utility libs/functional
    # Thread Related
    libs/thread
    libs/chrono libs/integer libs/container
    libs/ratio libs/intrusive libs/date_time
    libs/algorithm libs/rational libs/exception
    libs/array libs/lexical_cast libs/regex
    libs/range libs/unordered libs/numeric/conversion
    libs/tokenizer
    #
    libs/headers
    # I thought this was only for windows but ubuntu compile also required it
    libs/winapi
)

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
            # wich is bad. When you switch to release ti still installs
            # to debug. So we need to write these
            -DINSTALL_BIN_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_INC_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_LIB_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_MAN_DIR=${ZLIB_INSTALL_LOC}
            -DINSTALL_PKGCONFIG_DIR=${ZLIB_INSTALL_LOC}
            -DCMAKE_DEBUG_POSTFIX:STRING=d

)
append_cmake_prefix_path(${ZLIB_INSTALL_LOC})

# Boost FS and thread
set(BOOST_INSTALL_LOC ${OIIO_TMP_DIR}/boost)
mray_build_ext_dependency_git(
        NAME boost_fs_thrd_ext
        URL "https://github.com/boostorg/boost.git"
        TAG "b6928ae5c92e21a04bbe17a558e6e066dbe632f6" # v1.82.0
        # We will only download some of the boost
        SPECIFIC_SUBMODULES ${OIIO_BOOST_SUBMODULE_LIST}

        LICENSE_NAME "LICENSE_1_0.txt"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${BOOST_INSTALL_LOC}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DBOOST_INCLUDE_LIBRARIES:STRING=filesystem|thread
            -DBOOST_INSTALL_LAYOUT=tagged
)
append_cmake_prefix_path(${BOOST_INSTALL_LOC})

# OpenEXR & IMath
set(IMATH_INSTALL_LOC ${OIIO_TMP_DIR}/imath)
mray_build_ext_dependency_git(
        NAME imath_ext
        URL "https://github.com/AcademySoftwareFoundation/Imath.git"
        TAG "9fe5ed213cd99237976f2eb48b8bf86e112d90fa" # v3.1.10

        LICENSE_NAME "LICENSE.md"
        # OpenImageIO uses Imath as a subdirectory on includes.
        # Altough we do not use it, we need to install it to proper folder
        # DONT_OVERRIDE_INSTALL_SUFFIXES
        # OVERRIDE_INSTALL_PREFIX ${IMATH_INSTALL_LOC}

        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_TESTING=OFF
            -DCMAKE_DEBUG_POSTFIX:STRING=d
)
#append_cmake_prefix_path(${IMATH_INSTALL_LOC})

set(OPENEXR_INSTALL_LOC ${OIIO_TMP_DIR}/openexr)
mray_build_ext_dependency_git(
        NAME openexr_ext
        URL "https://github.com/AcademySoftwareFoundation/openexr.git"
        TAG "456bf42dc2fddb647c0e9ab911656e5cf757fc36" # v3.2.0

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
            -DZLIB_ROOT=${ZLIB_INSTALL_LOC}
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            #-DImath_DIR:PATH=${IMATH_INSTALL_LOC}/cmake/Imath

        DEPENDENCIES
            imath_ext
)
append_cmake_prefix_path(${OPENEXR_INSTALL_LOC})

# Lib TIFF
set(LIBTIFF_INSTALL_LOC ${OIIO_TMP_DIR}/libtiff)
mray_build_ext_dependency_git(
        NAME libtiff_ext
        URL "https://gitlab.com/libtiff/libtiff.git"
        TAG "38eb7b00cb5767770017fb91743a960ffd96d774" # v4.5.0

        LICENSE_NAME "LICENSE.md"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${LIBTIFF_INSTALL_LOC}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -Dtiff-tools=OFF
            -Dtiff-tests=OFF
            -Dtiff-docs=OFF
            -Dzlib=ON
            -DZLIB_ROOT:PATH=${ZLIB_INSTALL_LOC}
            -DJPEG_ROOT:PATH=${JPEGTURBO_INSTALL_LOC}
            -DCMAKE_DEBUG_POSTFIX:STRING=d
            # Finds stuff from miniconda since it is on path
            # It may be on path like in my system
            -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
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
)
append_cmake_prefix_path(${JPEGTURBO_INSTALL_LOC})

# Lib PNG
set(LIBPNG_INSTALL_LOC ${OIIO_TMP_DIR}/libPNG)
mray_build_ext_dependency_git(
        NAME libpng_ext
        URL "https://github.com/glennrp/libpng.git"
        TAG "07b8803110da160b158ebfef872627da6c85cbdf" # v2.1.5.1

        LICENSE_NAME "LICENSE"
        DONT_OVERRIDE_INSTALL_SUFFIXES
        OVERRIDE_INSTALL_PREFIX ${LIBPNG_INSTALL_LOC}
        BUILD_ARGS
            -DPNG_BUILD_ZLIB=OFF
            -DPNG_EXECUTABLES=OFF
            -DPNG_DEBUG=ON
            -DPNG_FAMEWORK=OFF
            -DPNG_SHARED=OFF
            -DPNG_STATIC=ON
            -DPNG_TESTS=OFF
            -DCMAKE_DEBUG_POSTFIX:STRING=d
        DEPENDENCIES
            oiio_zlib_ext
)
append_cmake_prefix_path(${LIBPNG_INSTALL_LOC})

# TODO: Add Jpeg2000, HEIF, HDR

# And finally oiio
mray_build_ext_dependency_git(
        NAME oiio_ext
        URL "https://github.com/OpenImageIO/oiio.git"
        TAG "c518e2da23cca3915beb70e7b7f2e0fd485af24d" # v2.5.7.0

        LICENSE_NAME "LICENSE.md"
        BUILD_ARGS
            # OIIO caches xxx_ROOT variables
            # and libarries does not change between configs

            # TODO: Change this later(oiio has "print" clashes with fmt)
            -DCMAKE_CXX_STANDARD=20
            -DBoost_USE_STATIC_LIBS=ON
            -DBoost_NO_WARN_NEW_VERSIONS=ON
            -DEMBEDPLUGINS=ON
            -DUSE_PYTHON=OFF
            -DUSE_OPENGL=ON
            -DUSE_QT=OFF
            -DBUILD_SHARED_LIBS=ON
            -DLINKSTATIC=ON
            -DOIIO_BUILD_TESTS=OFF
            -DOIIO_BUILD_TOOLS=OFF
            -DBUILD_DOCS=OFF
            -DBUILD_TESTING=OFF
            -DINSTALL_DOCS=OFF
            -DOIIO_DISABLE_BOOST_STACKTRACE=ON
            -DBUILD_MISSING_FMT=OFF # We have a fmt
            -DINTERNALIZE_FMT=OFF # On Linux this was required
            -DALWAYS_PREFER_CONFIG=ON
            # Do not use system
            -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            # Dep Root Files
            # AoS
            #-DImath_DIR:PATH=${OIIO_LIB_CONFIG_DIR}/cmake/Imath
            -DImath_ROOT:PATH=${IMATH_INSTALL_LOC}
            -DOpenEXR_ROOT:PATH=${OPENEXR_INSTALL_LOC}
            -DBoost_ROOT:PATH=${BOOST_INSTALL_LOC}
            -DZLIB_ROOT:PATH=${ZLIB_INSTALL_LOC}
            -DJPEGTurbo_ROOT:PATH=${JPEGTURBO_INSTALL_LOC}
            -DJPEG_ROOT:PATH=${JPEGTURBO_INSTALL_LOC}
            -DTIFF_ROOT:PATH=${LIBTIFF_INSTALL_LOC}
            -DPNG_ROOT:PATH=${LIBPNG_INSTALL_LOC}
            -DVERBOSE=ON

            # Do not install msvc runtime libraries
            -DCMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP=ON

        DEPENDENCIES
            boost_fs_thrd_ext
            openexr_ext
            libtiff_ext
            libjpeg_turbo_ext
            libpng_ext
            fmt_ext
)

# Issue of https://github.com/OpenImageIO/oiio/pull/3862
# It is not merged but we can patch it!
# I will add this as an extra step since we cant modify
# patch step after.
set(MRAY_OIIO_SRC_LOCATION ${MRAY_EXT_DIRECTORY}/oiio_ext/)

# DL'in file here instead
# file(DOWNLOAD https://github.com/OpenImageIO/oiio/pull/3862.patch
#               ${MRAY_OIIO_SRC_LOCATION}/../3862.patch)

# ExternalProject_Add_Step(oiio_ext patch_extra
#                 WORKING_DIRECTORY ${MRAY_OIIO_SRC_LOCATION}/
#                 # Couldnt able to make this work
#                 # COMMAND curl --create-dirs -s -o ${MRAY_OIIO_SRC_LOCATION}/3862.patch
#                 #         "https://github.com/OpenImageIO/oiio/pull/3862.patch"
#                 # Download patch at configure step, then copy here
#                 #
#                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                         ${MRAY_OIIO_SRC_LOCATION}/../3862.patch
#                         .
#                 # This is also not good, it will move head back and forth but this tecnhically
#                 # will run only initially so it is kinda fine.
#                 COMMAND git am 3862.patch
#                 DEPENDEES download update
#                 DEPENDERS configure
#                 COMMENT "Applying a patch to oiio"
# )

# Get the name of the deps outside
set(MRAY_OIIO_EXT_DEP_TARGETS
    ${MRAY_ALL_EXT_DEP_TARGETS} PARENT_SCOPE)