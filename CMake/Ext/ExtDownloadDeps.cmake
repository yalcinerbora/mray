
include(ExternalProject)

# Cosmetic wrapper of "ExternalProject_Add"
#   All libraries will be written on to Lib/$(Platform)/$(Config)
#   folders. (Config is not mandatory but it is neat while debugging)
#
#   Header files will be on Lib/include
#
# It is written wrt. to
# https://github.com/jeffamstutz/superbuild_ospray/blob/main/macros.cmake

macro(append_cmake_prefix_path)
  list(APPEND CMAKE_PREFIX_PATH ${ARGN})
  string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endmacro()

function(mray_build_ext_dependency_git)

    # Parse Args
    set(options SKIP_INSTALL DONT_OVERRIDE_INSTALL_SUFFIXES FORCE_RELEASE)
    set(oneValueArgs NAME URL TAG SOURCE_SUBDIR OVERRIDE_INSTALL_PREFIX LICENSE_NAME APPLY_PATCH)
    set(multiValueArgs BUILD_ARGS DEPENDENCIES SPECIFIC_SUBMODULES)

    cmake_parse_arguments(BUILD_SUBPROJECT "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(SUBPROJECT_PREFIX_DIR ${MRAY_PLATFORM_EXT_DIRECTORY})
    set(SUBPROJECT_BUILD_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/build/${BUILD_SUBPROJECT_NAME})
    set(SUBPROJECT_STMP_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/build/${BUILD_SUBPROJECT_NAME}-stamp)
    set(SUBPROJECT_TMP_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/build/${BUILD_SUBPROJECT_NAME}-tmp)
    set(SUBPROJECT_INSTALL_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/build/${BUILD_SUBPROJECT_NAME}-install)
    set(SUBPROJECT_LOG_DIR ${SUBPROJECT_STMP_DIR})
    set(SUBPROJECT_SRC_DIR ${MRAY_EXT_DIRECTORY}/${BUILD_SUBPROJECT_NAME})
    set(SUBPROJECT_DL_DIR ${SUBPROJECT_SRC_DIR})

    if(BUILD_SUBPROJECT_SKIP_INSTALL)
        set(SUBPROJECT_INSTALL_COMMAND_ARG "INSTALL_COMMAND")
        # Could not make the empty string work, so printing a message
        # as a "install" command
        set(SUBPROJECT_INSTALL_COMMAND
            ${CMAKE_COMMAND} -E echo "Skipping default install for ${BUILD_SUBPROJECT_NAME}")
    endif()

    # Override install prefix if requested
    # this install will be intermediate install
    # after this function user can copy the files using the appropirate
    # location
    if(NOT "${BUILD_SUBPROJECT_OVERRIDE_INSTALL_PREFIX}" STREQUAL "")
        set(SUBPROJECT_INSTALL_PREFIX ${BUILD_SUBPROJECT_OVERRIDE_INSTALL_PREFIX}/)
    else()
        set(SUBPROJECT_INSTALL_PREFIX ${MRAY_LIB_DIRECTORY}/)
    endif()

    # Check if specific submodules are requested
    if(BUILD_SUBPROJECT_SPECIFIC_SUBMODULES)
        list(PREPEND BUILD_SUBPROJECT_SPECIFIC_SUBMODULES "GIT_SUBMODULES")
    endif()

    # Principled install locations
    set(SUBPROJECT_INSTALL_SUFFIXES
        -DCMAKE_INSTALL_INCLUDEDIR:PATH=Include
        -DCMAKE_INSTALL_DOCDIR:PATH=Docs/${BUILD_SUBPROJECT_NAME}
        -DCMAKE_INSTALL_DATADIR:PATH=${MRAY_PLATFORM_NAME}/$<CONFIG>
        -DCMAKE_INSTALL_LIBDIR:PATH=${MRAY_PLATFORM_NAME}/$<CONFIG>
        -DCMAKE_INSTALL_BINDIR:PATH=${MRAY_PLATFORM_NAME}/$<CONFIG>
        -DCMAKE_INSTALL_DATAROOTDIR:PATH=${MRAY_PLATFORM_NAME}/$<CONFIG>
    )
    #string(REPLACE ";" "|" SUBPROJECT_INSTALL_SUFFIXES "${SUBPROJECT_INSTALL_SUFFIXES}")
    if(BUILD_SUBPROJECT_DONT_OVERRIDE_INSTALL_SUFFIXES)
        set(SUBPROJECT_INSTALL_SUFFIXES)
    endif()

    if(BUILD_SUBPROJECT_APPLY_PATCH)
        set(BUILD_SUBPROJECT_APPLY_PATCH
            PATCH_COMMAND
                curl ${BUILD_SUBPROJECT_APPLY_PATCH} -s -o ${BUILD_SUBPROJECT_NAME}.patch
            # Weird but what to do ...
            # Reset the patch and apply it again
            COMMAND
                git checkout -- .
            COMMAND
                git apply -v ${BUILD_SUBPROJECT_NAME}.patch
        )
    endif()

    # Actual Call
    ExternalProject_Add(${BUILD_SUBPROJECT_NAME}
        PREFIX ${SUBPROJECT_PREFIX_DIR}
        BINARY_DIR ${SUBPROJECT_BUILD_DIR}
        TMP_DIR ${SUBPROJECT_TMP_DIR}
        SOURCE_DIR ${SUBPROJECT_SRC_DIR}
        LOG_DIR ${SUBPROJECT_LOG_DIR}
        INSTALL_DIR ${SUBPROJECT_INSTALL_DIR}
        STAMP_DIR ${SUBPROJECT_STMP_DIR}
        DOWNLOAD_DIR ${SUBPROJECT_DL_DIR}

        BUILD_IN_SOURCE OFF

        ${BUILD_SUBPROJECT_APPLY_PATCH}

        # DL Repo
        GIT_REPOSITORY ${BUILD_SUBPROJECT_URL}
        GIT_TAG ${BUILD_SUBPROJECT_TAG}
        #GIT_SHALLOW ON
        # with specific submodules if requested
        ${BUILD_SUBPROJECT_SPECIFIC_SUBMODULES}

        # Custom build root location if required
        SOURCE_SUBDIR ${BUILD_SUBPROJECT_SOURCE_SUBDIR}

        # In order to skip install
        # I could not get it to work with lists dunno why
        ${SUBPROJECT_INSTALL_COMMAND_ARG}
        ${SUBPROJECT_INSTALL_COMMAND}

        # Log the outputs instead of printing
        # except when there is an error
        LOG_DOWNLOAD OFF
        LOG_UPDATE ON
        LOG_PATCH ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_INSTALL ON
        LOG_OUTPUT_ON_FAILURE ON

        # Common args (it will share the generator and compiler)
        LIST_SEPARATOR | # Use the alternate list separator
        CMAKE_ARGS
            #-DCMAKE_BUILD_TYPE:STRING=$<CONFIG>
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
            -DCMAKE_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
            # Do not use system libraries
            -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
            # Mandate a debug postfix
            -DCMAKE_DEBUG_POSTFIX=d

            # TODO: Do not do fresh install
            # find out a way to change package locations
            # since the very first config's dependencies
            # stays in the cache
            --fresh

            # Install Stuff
            -DCMAKE_INSTALL_PREFIX:PATH=${SUBPROJECT_INSTALL_PREFIX}
            ${SUBPROJECT_INSTALL_SUFFIXES}
            # Extra args from user to pass CMake
            ${BUILD_SUBPROJECT_BUILD_ARGS}



        BUILD_ALWAYS OFF
    )

    # Copy license file if available to the main Lib directory
    string(REPLACE "_ext" "" BUILD_SUBPROJECT_NAME_NO_EXT ${BUILD_SUBPROJECT_NAME})

    if(NOT ${BUILD_SUBPROJECT_LICENSE_NAME} STREQUAL "")
        ExternalProject_Add_Step(${BUILD_SUBPROJECT_NAME} copy_license
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SUBPROJECT_SRC_DIR}/${BUILD_SUBPROJECT_LICENSE_NAME}
                ${MRAY_LIB_DIRECTORY}/${BUILD_SUBPROJECT_NAME_NO_EXT}_LICENSE
                DEPENDEES download update patch)
    endif()

    if(BUILD_SUBPROJECT_DEPENDENCIES)
        ExternalProject_Add_StepDependencies(${BUILD_SUBPROJECT_NAME}
                                             "install" ${BUILD_SUBPROJECT_DEPENDENCIES})
    endif()

# Get the name of the deps outside
set(MRAY_ALL_EXT_DEP_TARGETS ${MRAY_ALL_EXT_DEP_TARGETS}
    ${BUILD_SUBPROJECT_NAME} PARENT_SCOPE)

endfunction()