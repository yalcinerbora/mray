
set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/SpectraLutGen)

# We may compile this with O3 for all configurations so we do
# not want any dependencies (due to MSVC ITERATOR_DEBUG_LEVEL stuff)
# so we compile some translation units for this executable here as well
set(CORE_LIB_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Core)

set(SRC_ALL ${CURRENT_SOURCE_DIR}/main.cpp
            ${CORE_LIB_SOURCE_DIR}/ColorFunctions.cpp)

source_group("" FILES ${SRC_ALL})

add_executable(SpectraLUTGen ${SRC_ALL})
target_link_libraries(SpectraLUTGen)

# Now we do not want this to run in debug mode even in debug mode
# but we want to debug this while developing etc.
# (Debug build completes 64^3 tables in couple minutes instead of ~10 seconds).
# There is no straightforward way to do this in CMAKE afaik.
#
# So we can make this project external and load it via cmake functionality (kinda weird)
# or add -O3 here for both debug and release and when there is a bug we can adjust the code below
#
# TODO: Search for the specific use case in other projects/internet etc. there may be a better way to do this
#
# ================ #
#      START       #
# ================ #
# Our generic options (for debug/release)
# target_link_libraries(SpectraLUTGen
#                       PRIVATE
#                       mray::meta_compile_opts)
#
# ===================================================== #
# Either comment this and uncomment code or vice versa. #
# ===================================================== #
# Basic compile options for both debug and release
if(MSVC)
    set(COMPILE_OPTIONS
        /Zc:inline
        /Zc:__cplusplus
        /permissive-
        /MD /O2 /Oi /Zi /Ob3 /DNDEBUG
        /EHsc
        /DMRAY_MSVC)
else()
    set(COMPILE_OPTIONS -O3 -g -fno-math-errno)
endif()

if(MSVC)
    target_compile_definitions(SpectraLUTGen PRIVATE MRAY_MSVC)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_definitions(SpectraLUTGen PRIVATE MRAY_CLANG)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*GNU")
    target_compile_definitions(SpectraLUTGen PRIVATE MRAY_GCC)
endif()

target_include_directories(SpectraLUTGen
                           PRIVATE ${MRAY_SOURCE_DIRECTORY}
                           SYSTEM PRIVATE ${MRAY_LIB_INCLUDE_DIRECTORY})
target_compile_options(SpectraLUTGen PRIVATE ${COMPILE_OPTIONS})
# ================ #
#       END        #
# ================ #

# Debug Parameters (For my own use case)
if(MSVC)
    set(DEBUG_ARGS "")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "64 ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "ACES2065_1 ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} ". ")

    set_target_properties(SpectraLUTGen PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()

set_target_properties(SpectraLUTGen PROPERTIES FOLDER ${MRAY_PREDEFINED_TARGETS_FOLDER})

# ======================================= #
#   Target that builds spectra LUT data   #
# ======================================= #
function(spectra_gen_command)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs)
    cmake_parse_arguments(SPECTRA_GEN_COMMAND
                          "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})


    set(FILE_NAME "${SPECTRA_GEN_COMMAND_NAME}.mrspectra")
    set(OUTPUT_PATH "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/${FILE_NAME}")
    add_custom_command(OUTPUT ${OUTPUT_PATH}
                       COMMAND SpectraLUTGen 64 ${SPECTRA_GEN_COMMAND_NAME} ${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/
                       COMMENT "Generating Spectra LUT for ${SPECTRA_GEN_COMMAND_NAME}..."
                       DEPENDS SpectraLUTGen
                       VERBATIM)

endfunction()

spectra_gen_command(NAME "ACES2065_1")
spectra_gen_command(NAME "ACES_CG")
spectra_gen_command(NAME "REC_709")
spectra_gen_command(NAME "REC_2020")
spectra_gen_command(NAME "DCI_P3")
spectra_gen_command(NAME "ADOBE_RGB")

add_custom_target(SpectraLUTGen_AllCS
                  DEPENDS
                  "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/ACES2065_1.mrspectra"
                  "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/ACES_CG.mrspectra"
                  "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/REC_709.mrspectra"
                  "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/REC_2020.mrspectra"
                  "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/DCI_P3.mrspectra"
                  "${MRAY_CONFIG_BIN_DIRECTORY}/SpectraLUT/ADOBE_RGB.mrspectra")

# User needs to call this explicitly
set_target_properties(SpectraLUTGen_AllCS PROPERTIES FOLDER ${MRAY_PREDEFINED_TARGETS_FOLDER})
set_target_properties(SpectraLUTGen_AllCS PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(SpectraLUTGen_AllCS PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)