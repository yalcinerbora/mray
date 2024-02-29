
# Configure Time check the availablity of the libraries
# that is build by "Ext" project
#
# TODO: Now this code is for development purposes
# On config time it builds both debug and release builds of the libraries
# However maybe for install etc user wants a single config of the MRay
# thus we only need to build that config for the external libraries too.
# Do an adjustment for that...

set(MRAY_EXT_DIR ${MRAY_DIRECTORY}Ext/${MRAY_PLATFORM_NAME})
set(MRAY_CMAKE_EXT_DIR ${MRAY_DIRECTORY}CMake/Ext)
set(MRAY_EXT_BUILD_DIR ${MRAY_EXT_DIR}/build-CMake)
set(MRAY_EXT_CONFIG_META_ARGS
        -S ${MRAY_CMAKE_EXT_DIR}
        -B ${MRAY_EXT_BUILD_DIR}
        -G ${CMAKE_GENERATOR}
        # Main Config Options
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DMRAY_BUILD_TESTS=${MRAY_BUILD_TESTS}
)


set(MRAY_EXTERNAL_LIBRARIES_STAMP ${MRAY_CONFIG_LIB_DIRECTORY}/MRayLib.stamp)

set(MRAY_EXTERNAL_LIBRARIES
    ${MRAY_CONFIG_LIB_DIRECTORY}/fmt${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${MRAY_LIB_INCLUDE_DIRECTORY}/nlohmann/json.hpp
    ${MRAY_LIB_INCLUDE_DIRECTORY}/nlohmann/json_fwd.hpp
    ${MRAY_LIB_INCLUDE_DIRECTORY}/BS/BS_thread_pool.hpp
    ${MRAY_CONFIG_LIB_DIRECTORY}/assimp$<$<CONFIG:Debug>:d>
    ${MRAY_CONFIG_LIB_DIRECTORY}/GFGFileIO$<$<CONFIG:Debug>:d>

    # ${MRAY_CONFIG_LIB_DIRECTORY}/ZLIB$<$<CONFIG:Debug>:d>
    # ${MRAY_CONFIG_LIB_DIRECTORY}/spdlog$<$<CONFIG:Debug>:d>${CMAKE_SHARED_LIBRARY_SUFFIX}
    # ${MRAY_CONFIG_LIB_DIRECTORY}/glbinding$<$<CONFIG:Debug>:d>
    # ${MRAY_CONFIG_LIB_DIRECTORY}/glfw3$<$<CONFIG:Debug>:d>dll
    # ${MRAY_CONFIG_LIB_DIRECTORY}/imgui$<$<CONFIG:Debug>:d>
    # ${MRAY_CONFIG_LIB_DIRECTORY}/imgui_tex_inspect$<$<CONFIG:Debug>:d>
    # ${MRAY_CONFIG_LIB_DIRECTORY}/OpenImageIO$<$<CONFIG:Debug>:d>${CMAKE_SHARED_LIBRARY_SUFFIX}
    # ${MRAY_CONFIG_LIB_DIRECTORY}/OpenImageIO_Util$<$<CONFIG:Debug>:d>${CMAKE_SHARED_LIBRARY_SUFFIX}
)

if(MRAY_BUILD_TESTS)
    set(MRAY_EXTERNAL_LIBRARIES ${MRAY_EXTERNAL_LIBRARIES}
        ${MRAY_CONFIG_LIB_DIRECTORY}/gtest${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${MRAY_CONFIG_LIB_DIRECTORY}/gmock${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
endif()

# Add a custom command for external library generation
get_property(MRAY_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(MRAY_IS_MULTI_CONFIG)
    add_custom_command(OUTPUT ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMAND ${CMAKE_COMMAND} ${MRAY_EXT_CONFIG_META_ARGS}
        COMMAND ${CMAKE_COMMAND} --build ${MRAY_EXT_BUILD_DIR} --config $<IF:$<CONFIG:ReleaseDBG>,Release,$<CONFIG>>
        COMMAND ${CMAKE_COMMAND} -E touch ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMENT "Compiling external libraries($<CONFIG>). This may take a while..."
    )
else()
    add_custom_command(OUTPUT ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMAND ${CMAKE_COMMAND} ${MRAY_EXT_CONFIG_META_ARGS} -DCMAKE_BUILD_TYPE=$<IF:$<CONFIG:ReleaseDBG>,Release,$<CONFIG>>
        COMMAND ${CMAKE_COMMAND} --build ${MRAY_EXT_BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMENT "Compiling external libraries($<CONFIG>). This may take a while..."
    )
endif()

add_custom_target(MRayExternal ALL DEPENDS ${MRAY_EXTERNAL_LIBRARIES_STAMP})
set_target_properties(MRayExternal PROPERTIES FOLDER ${MRAY_PREDEFINED_TARGETS_FOLDER})

#  Add a custom command on mvsc to dll copy

# Tests
# On windows we need to copy the dlls to the binary folder
if(MSVC)
    # add_custom_command(TARGET MRayExternal POST_BUILD
    #                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #                    "${MRAY_CONFIG_LIB_DIRECTORY}/spdlog$<$<CONFIG:Debug>:d>.dll"
    #                    "${MRAY_CONFIG_LIB_DIRECTORY}/ZLIB$<$<CONFIG:Debug>:d>1.dll"
    #                    ${MRAY_CONFIG_BIN_DIRECTORY})

    # Extra DLL copy for tests
    if(${MRAY_BUILD_TESTS})
        add_custom_command(TARGET MRayExternal POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gtest_main.dll"
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gmock_main.dll"
                           ${MRAY_CONFIG_BIN_DIRECTORY})
    endif()
endif()

# Everything should be fine now
# Our directory structure is solid
# I've tried find_package route but it is messy :(
# install config location is not standart for CMake
# or the standart is not compatible for multi-configuration
# structure of arrays style of directory layout
#
# We create imported targets by hand
# everything should be compiled properly
# Some include nuances exists but we will handle it here
#
# TODO: Change this maybe later?
function(gen_imported_target)
    # Parse Args
    set(options STATIC SHARED)
    set(oneValueArgs NAME DEBUG_TARGET RELEASE_TARGET)
    set(multiValueArgs EXTRA_INCLUDES EXTRA_DEFINES)
    cmake_parse_arguments(GEN_IMPORTED_TGT "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

    if(GEN_IMPORTED_TGT_STATIC AND GEN_IMPORTED_TGT_SHARED)
        message(FATAL_ERROR "gen_imported_target called with both SHARED and STATIC")
        return()
    endif()
    if(NOT GEN_IMPORTED_TGT_STATIC AND NOT GEN_IMPORTED_TGT_SHARED)
        message(FATAL_ERROR "gen_imported_target must be called with either SHARED or STATIC")
        return()
    endif()

    # Determine the naming stuff
    if(GEN_IMPORTED_TGT_STATIC)
        set(TGT_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
        set(TGT_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    else()
        set(TGT_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(TGT_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    endif()

    if(GEN_IMPORTED_TGT_STATIC)
        add_library(${GEN_IMPORTED_TGT_NAME} STATIC IMPORTED)
    else()
        add_library(${GEN_IMPORTED_TGT_NAME} SHARED IMPORTED)
    endif()
    set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES IMPORTED_GLOBAL TRUE)

    set(DBG_NAME "${TGT_PREFIX}${GEN_IMPORTED_TGT_DEBUG_TARGET}")
    set(REL_NAME "${TGT_PREFIX}${GEN_IMPORTED_TGT_RELEASE_TARGET}")

    set_property(TARGET ${GEN_IMPORTED_TGT_NAME} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_property(TARGET ${GEN_IMPORTED_TGT_NAME} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

    set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
        IMPORTED_LOCATION_DEBUG "${MRAY_PLATFORM_LIB_DIRECTORY}/Debug/${DBG_NAME}${TGT_SUFFIX}"
        IMPORTED_LOCATION_RELEASE "${MRAY_PLATFORM_LIB_DIRECTORY}/Release/${REL_NAME}${TGT_SUFFIX}"
    )
    set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES MAP_IMPORTED_CONFIG_RELEASEDBG Release)

    if(GEN_IMPORTED_TGT_EXTRA_INCLUDES)
        target_include_directories(${GEN_IMPORTED_TGT_NAME}
                                    INTERFACE
                                    ${GEN_IMPORTED_TGT_EXTRA_INCLUDES})
    endif()

    if(GEN_IMPORTED_TGT_EXTRA_DEFINES)
        target_compile_definitions(${GEN_IMPORTED_TGT_NAME}
                                    INTERFACE
                                    ${GEN_IMPORTED_TGT_EXTRA_DEFINES})
    endif()

    if(MSVC AND NOT GEN_IMPORTED_TGT_STATIC)
        set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
                              IMPORTED_IMPLIB_DEBUG "${MRAY_PLATFORM_LIB_DIRECTORY}/Debug/${DBG_NAME}.lib"
                              IMPORTED_IMPLIB_RELEASE "${MRAY_PLATFORM_LIB_DIRECTORY}/Release/${REL_NAME}.lib")
    endif()

endfunction()

# Header only libraries
# Create mock targets, project layout enables direct access to the headers
# These are generated for verbosity
add_library(json INTERFACE)
target_compile_definitions(json INTERFACE JSON_DIAGNOSTICS=1)
add_library(nlohmann::json ALIAS json)
add_library(thread_pool INTERFACE)
add_library(bs::thread_pool ALIAS thread_pool)

gen_imported_target(NAME fmt::fmt STATIC DEBUG_TARGET fmt RELEASE_TARGET fmt
                    EXTRA_DEFINES FMT_USE_NONTYPE_TEMPLATE_ARGS=0)
gen_imported_target(NAME simdjson::simdjson STATIC DEBUG_TARGET simdjsond RELEASE_TARGET simdjson)
gen_imported_target(NAME assimp::assimp SHARED DEBUG_TARGET assimpd RELEASE_TARGET assimp)
gen_imported_target(NAME gfgfileio::gfgfileio STATIC DEBUG_TARGET GFGFileIOd RELEASE_TARGET GFGFileIO)
gen_imported_target(NAME oiio::oiio SHARED DEBUG_TARGET OpenImageIOd RELEASE_TARGET OpenImageIO)
gen_imported_target(NAME oiio::oiio_util SHARED DEBUG_TARGET OpenImageIO_Utild RELEASE_TARGET OpenImageIO_Util)

# gen_imported_target(NAME spdlog::spdlog SHARED DEBUG_TARGET spdlogd RELEASE_TARGET spdlog)
# gen_imported_target(NAME glbinding::glbinding SHARED DEBUG_TARGET glbindingd RELEASE_TARGET glbinding
#                     EXTRA_INCLUDES ${MRAY_LIB_INCLUDE_DIRECTORY}/glbinding/3rdparty)
# gen_imported_target(NAME glfw3::glfw3 SHARED DEBUG_TARGET glfw3ddll RELEASE_TARGET glfw3dll)
# gen_imported_target(NAME imgui::imgui STATIC DEBUG_TARGET imguid RELEASE_TARGET imgui)
# gen_imported_target(NAME imgui::imgui_ti STATIC DEBUG_TARGET imgui_tex_inspectd RELEASE_TARGET imgui_tex_inspect)

#gen_imported_target(NAME ZLIB::ZLIB SHARED DEBUG_TARGET ZLIBd RELEASE_TARGET ZLIB)

if(MRAY_BUILD_TESTS)
    gen_imported_target(NAME gtest::gtest SHARED DEBUG_TARGET gtest RELEASE_TARGET gtest)
    gen_imported_target(NAME gtest::gtest_main SHARED DEBUG_TARGET gtest_main RELEASE_TARGET gtest_main)
    gen_imported_target(NAME gtest::gmock SHARED DEBUG_TARGET gmock RELEASE_TARGET gmock)
    gen_imported_target(NAME gtest::gmock_main SHARED DEBUG_TARGET gmock_main RELEASE_TARGET gmock_main)
endif()
