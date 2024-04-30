macro(populate_mray_directory_variables)

    if(NOT MRAY_PLATFORM_NAME)
        message (FATAL_ERROR "Please set MRAY_PLATFORM_NAME before
                            calling populate_mray_directory_variables macro.")
        return()
    endif()

    set(MRAY_SOURCE_DIRECTORY ${ARGV0}Source)
    set(MRAY_LIB_DIRECTORY ${ARGV0}Lib)
    set(MRAY_LIB_INCLUDE_DIRECTORY ${ARGV0}Lib/Include)
    set(MRAY_CONFIG_LIB_DOC_DIRECTORY ${ARGV0}Lib/Docs)
    set(MRAY_BIN_DIRECTORY ${ARGV0}Bin)
    set(MRAY_EXT_DIRECTORY ${ARGV0}Ext)
    set(MRAY_RESOURCE_DIRECTORY ${ARGV0}Resources)
    # Working Dir is used for debugging (currently shaders are here so...)
    # For Visual Studio Projects this is copied to WorkingDir property for executables
    set(MRAY_WORKING_DIRECTORY ${ARGV0}WorkingDir)

    # Platform Specific Lib Bin Ext Folders
    set(MRAY_PLATFORM_LIB_DIRECTORY ${MRAY_LIB_DIRECTORY}/${MRAY_PLATFORM_NAME})
    set(MRAY_PLATFORM_BIN_DIRECTORY ${MRAY_BIN_DIRECTORY}/${MRAY_PLATFORM_NAME})
    set(MRAY_PLATFORM_EXT_DIRECTORY ${MRAY_EXT_DIRECTORY}/${MRAY_PLATFORM_NAME})
    # Platform & Configurations Related Lib Bin folders
    set(MRAY_CONFIG_LIB_DIRECTORY ${MRAY_PLATFORM_LIB_DIRECTORY}/$<IF:$<CONFIG:SanitizeR>,Release,$<CONFIG>>)
    set(MRAY_CONFIG_BIN_DIRECTORY ${MRAY_PLATFORM_BIN_DIRECTORY}/$<CONFIG>)
    # Set cmake vars for output
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})

endmacro()

macro(generate_platform_name)

    # Determine Platform and Config
    # Only Windows (and probably not mingw adn cygwin) and Linux is supported
    if(MSVC)
        set(MRAY_PLATFORM_NAME Win)
    elseif(UNIX AND NOT APPLE)
        set(MRAY_PLATFORM_NAME Linux)
    else()
        message(FATAL_ERROR "Unknown platform... Terminating CMake.")
        return()
    endif()

endmacro()