
# Configure Time check the availablity of the libraries
# that is build by "Ext" project
#
# TODO: Now this code is for development purposes
# On config time it builds both debug and release builds of the libraries
# However maybe for install etc user wants a single config of the MRay
# thus we only need to build that config for the external libraries too.
# Do an adjustment for that...

# Check if we should DL Embree
set(DMRAY_ENABLE_EMBREE OFF)
if(MRAY_BUILD_HOST_BACKEND AND MRAY_ENABLE_HW_ACCELERATION)
    set(DMRAY_ENABLE_EMBREE ON)
endif()

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
        -DMRAY_BUILD_VISOR=${MRAY_BUILD_VISOR}
        -DMRAY_BUILD_USD=${MRAY_ENABLE_USD}
        -DMRAY_DL_EMBREE=${DMRAY_ENABLE_EMBREE}
)

set(MRAY_EXTERNAL_LIBRARIES_STAMP ${MRAY_CONFIG_LIB_DIRECTORY}/MRayLib.stamp)

set(MRAY_EXTERNAL_LIBRARIES
    ${MRAY_CONFIG_LIB_DIRECTORY}/fmt${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${MRAY_LIB_INCLUDE_DIRECTORY}/nlohmann/json.hpp
    ${MRAY_LIB_INCLUDE_DIRECTORY}/nlohmann/json_fwd.hpp
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
        COMMAND ${CMAKE_COMMAND} --build ${MRAY_EXT_BUILD_DIR} -j --config $<IF:$<CONFIG:SanitizeR>,Release,$<CONFIG>>
        COMMAND ${CMAKE_COMMAND} -E touch ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMENT "Compiling external libraries($<CONFIG>). This may take a while..."
        USES_TERMINAL)
else()
    add_custom_command(OUTPUT ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMAND ${CMAKE_COMMAND} ${MRAY_EXT_CONFIG_META_ARGS} -DCMAKE_BUILD_TYPE=$<IF:$<CONFIG:SanitizeR>,Release,$<CONFIG>>
        COMMAND ${CMAKE_COMMAND} --build ${MRAY_EXT_BUILD_DIR} -j
        COMMAND ${CMAKE_COMMAND} -E touch ${MRAY_EXTERNAL_LIBRARIES_STAMP}
        COMMENT "Compiling external libraries($<CONFIG>). This may take a while..."
        USES_TERMINAL)
endif()

add_custom_target(MRayExternal DEPENDS ${MRAY_EXTERNAL_LIBRARIES_STAMP})
set_target_properties(MRayExternal PROPERTIES FOLDER ${MRAY_PREDEFINED_TARGETS_FOLDER})
set_target_properties(MRayExternal PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(MRayExternal PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)

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
    set(oneValueArgs NAME DEBUG_TARGET RELEASE_TARGET OVERRIDE_LIB_PREFIX VERSION)
    set(multiValueArgs EXTRA_INCLUDES EXTRA_DEFINES EXTRA_LIB_FOLDERS)
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

    set(GEN_IMPORTED_TGT_DBG_IMPORT_NAME ${DBG_NAME}${TGT_SUFFIX})
    set(GEN_IMPORTED_TGT_REL_IMPORT_NAME ${REL_NAME}${TGT_SUFFIX})
    # On linux, give the version to the link name, we will copy these if
    # these are shared libraries (Simulating Windows DLL search)
    if(UNIX AND NOT APPLE AND GEN_IMPORTED_TGT_SHARED AND GEN_IMPORTED_TGT_VERSION)
        set(GEN_IMPORTED_TGT_DBG_IMPORT_NAME ${DBG_NAME}${TGT_SUFFIX}.${GEN_IMPORTED_TGT_VERSION})
        set(GEN_IMPORTED_TGT_REL_IMPORT_NAME ${REL_NAME}${TGT_SUFFIX}.${GEN_IMPORTED_TGT_VERSION})
    endif()

    # TODO: GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX is not generic enough?
    # What if the system has per-config lib folders?
    if(GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX)
        set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
            IMPORTED_LOCATION_DEBUG "${GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX}/${GEN_IMPORTED_TGT_DBG_IMPORT_NAME}"
            IMPORTED_LOCATION_RELEASE "${GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX}/${GEN_IMPORTED_TGT_REL_IMPORT_NAME}")
    else()
        set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
            IMPORTED_LOCATION_DEBUG "${MRAY_PLATFORM_LIB_DIRECTORY}/Debug/${GEN_IMPORTED_TGT_DBG_IMPORT_NAME}"
            IMPORTED_LOCATION_RELEASE "${MRAY_PLATFORM_LIB_DIRECTORY}/Release/${GEN_IMPORTED_TGT_REL_IMPORT_NAME}")
    endif()

    # Default is to link with release configuration
    # since the build is "Release". You can change this to debug here.
    # This should only be valid in clang/gcc STL libraries, MSVC has "ITERATOR_DEBUG_LEVEL"
    # macros and other stuff that prevents linking Release/Debug builds. Thus, this is not propagated
    # as an option to the user.
    set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES MAP_IMPORTED_CONFIG_SANITIZER Release)
    #set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES MAP_IMPORTED_CONFIG_SANITIZER Debug)

    if(GEN_IMPORTED_TGT_EXTRA_INCLUDES)
        # Create these directories if they do not exist.
        # "MRayExternal" target did not run yet, so these may not be available
        file(MAKE_DIRECTORY ${GEN_IMPORTED_TGT_EXTRA_INCLUDES})
        target_include_directories(${GEN_IMPORTED_TGT_NAME}
                                    INTERFACE
                                    ${GEN_IMPORTED_TGT_EXTRA_INCLUDES})
    endif()

    if(GEN_IMPORTED_TGT_EXTRA_DEFINES)
        target_compile_definitions(${GEN_IMPORTED_TGT_NAME}
                                    INTERFACE
                                    ${GEN_IMPORTED_TGT_EXTRA_DEFINES})
    endif()

    if(GEN_IMPORTED_TGT_EXTRA_LIB_FOLDERS)
        target_link_directories(${GEN_IMPORTED_TGT_NAME}
                                INTERFACE
                                ${GEN_IMPORTED_TGT_EXTRA_LIB_FOLDERS})
    endif()

    if(WIN32 AND NOT GEN_IMPORTED_TGT_STATIC)
        # TODO: GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX is not generic enough?
        # What if the system has per-config lib folders?
        if(GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX)
            set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
                IMPORTED_IMPLIB_DEBUG "${GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX}/${DBG_NAME}.lib"
                IMPORTED_IMPLIB_RELEASE "${GEN_IMPORTED_TGT_OVERRIDE_LIB_PREFIX}/${REL_NAME}.lib")
        else()
            set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
                IMPORTED_IMPLIB_DEBUG "${MRAY_PLATFORM_LIB_DIRECTORY}/Debug/${DBG_NAME}.lib"
                IMPORTED_IMPLIB_RELEASE "${MRAY_PLATFORM_LIB_DIRECTORY}/Release/${REL_NAME}.lib")
        endif()
    endif()

endfunction()

function(gen_archive_pack)
    set(options HAS_DEBUG_SUFFIX PACKAGE_WHOLE_ARCHIVE)
    set(oneValueArgs NAME)
    set(multiValueArgs ARCHIVES EXTRA_LIBS EXTRA_INCLUDES EXTRA_DEFINES EXTRA_LIB_FOLDERS)
    cmake_parse_arguments(GEN_ARCHIVE_PACK "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

    add_library(${GEN_ARCHIVE_PACK_NAME} INTERFACE IMPORTED)
    set_target_properties(${GEN_ARCHIVE_PACK_NAME} PROPERTIES IMPORTED_GLOBAL TRUE)

    if(UNIX AND NOT APPLE)
        list(TRANSFORM GEN_ARCHIVE_PACK_ARCHIVES PREPEND ${CMAKE_STATIC_LIBRARY_PREFIX})
    endif()
    # Debug
    set(GEN_ARCHIVE_PACK_DBG_NAMES ${GEN_ARCHIVE_PACK_ARCHIVES})
    if(GEN_ARCHIVE_PACK_HAS_DEBUG_SUFFIX)
        list(TRANSFORM GEN_ARCHIVE_PACK_DBG_NAMES APPEND d)
    endif()
    list(TRANSFORM GEN_ARCHIVE_PACK_DBG_NAMES APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
    # Release
    set(GEN_ARCHIVE_PACK_REL_NAMES ${GEN_ARCHIVE_PACK_ARCHIVES})
    list(TRANSFORM GEN_ARCHIVE_PACK_REL_NAMES APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
    # Make it absolute path (kinda)

    # Add extra includes if needed
    if(GEN_ARCHIVE_PACK_EXTRA_INCLUDES)
        # Create these directories if they do not exist.
        # "MRayExternal" target did not run yet, so these may not be available
        file(MAKE_DIRECTORY ${GEN_ARCHIVE_PACK_EXTRA_INCLUDES})
        target_include_directories(${GEN_ARCHIVE_PACK_NAME}
                                    # Adding system since USD has many
                                    # warnings...
                                    SYSTEM INTERFACE
                                    ${GEN_ARCHIVE_PACK_EXTRA_INCLUDES})
    endif()
    if(GEN_ARCHIVE_PACK_EXTRA_DEFINES)
        target_compile_definitions(${GEN_ARCHIVE_PACK_NAME}
                                    INTERFACE
                                    ${GEN_ARCHIVE_PACK_EXTRA_DEFINES})
    endif()
    if(GEN_ARCHIVE_PACK_EXTRA_LIB_FOLDERS)
        target_link_directories(${GEN_ARCHIVE_PACK_NAME}
                                INTERFACE
                                ${GEN_ARCHIVE_PACK_EXTRA_LIB_FOLDERS})
    endif()

    # Convert to "whole archive" if requested
    if(GEN_ARCHIVE_PACK_PACKAGE_WHOLE_ARCHIVE)

        set(GEN_ARCHIVE_PACK_DBG_LINK_OPTS ${GEN_ARCHIVE_PACK_DBG_NAMES})
        set(GEN_ARCHIVE_PACK_REL_LINK_OPTS ${GEN_ARCHIVE_PACK_REL_NAMES})
        list(JOIN GEN_ARCHIVE_PACK_DBG_NAMES "," GEN_ARCHIVE_PACK_DBG_NAMES)
        string(PREPEND GEN_ARCHIVE_PACK_DBG_NAMES $<LINK_LIBRARY:WHOLE_ARCHIVE,)
        string(APPEND GEN_ARCHIVE_PACK_DBG_NAMES >)

        list(JOIN GEN_ARCHIVE_PACK_REL_NAMES "," GEN_ARCHIVE_PACK_REL_NAMES)
        string(PREPEND GEN_ARCHIVE_PACK_REL_NAMES $<LINK_LIBRARY:WHOLE_ARCHIVE,)
        string(APPEND GEN_ARCHIVE_PACK_REL_NAMES >)

    endif()
    # Finally the link command...
    target_link_libraries(${GEN_ARCHIVE_PACK_NAME}
                          INTERFACE
                          $<$<CONFIG:Debug>:${GEN_ARCHIVE_PACK_DBG_NAMES}>
                          $<$<CONFIG:Release>:${GEN_ARCHIVE_PACK_REL_NAMES}>
                          $<$<CONFIG:SanitizeR>:${GEN_ARCHIVE_PACK_REL_NAMES}>)

endfunction()

# Find OptiX if requested
if(MRAY_ENABLE_HW_ACCELERATION AND
   (MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")
   AND NOT MRAY_DISABLE_DEVICE_BACKEND)

    find_package(OPTIX 8)
    if(NOT OPTIX_FOUND)
        # Maybe we could not able to find the optix
        # expose the OPTIX_DIR (in cmake-GUI) so user can manually set it
        message(FATAL_ERROR "OptiX Not Found, either set OPTIX_INSTALL_DIR or \
                            disable MRAY_ENABLE_HW_ACCELERATION")
        set(MRAY_ENABLE_HW_ACCELERATION CACHE BOOL OFF)
    endif()

    # OptiX only needs include_dir since it automagically links with the driver
    add_library(optix INTERFACE)
    target_include_directories(optix INTERFACE ${OPTIX_INCLUDE_DIR})
    add_library(optix::optix ALIAS optix)
endif()

# Create custom embree target if requested
if(MRAY_ENABLE_HW_ACCELERATION AND MRAY_BUILD_HOST_BACKEND)
    gen_imported_target(NAME embree4::embree4_cpu SHARED
                        OVERRIDE_LIB_PREFIX ${MRAY_PLATFORM_LIB_DIRECTORY}/embree/lib
                        DEBUG_TARGET embree4
                        RELEASE_TARGET embree4
                        EXTRA_DEFINES
                            EMBREE_VERSION=4.4.0
                            EMBREE_VERSION_MAJOR=4
                            EMBREE_VERSION_MINOR=4
                            EMBREE_VERSION_PATCH=0
                            EMBREE_VERSION_NOTE=""
                            EMBREE_BUILD_TYPE=Release
                            EMBREE_TASKING_SYSTEM=TBB
                            EMBREE_TBB_COMPONENT=tbb
                            #
                            EMBREE_MAX_ISA=AVX512
                            EMBREE_ISA_SSE2
                            EMBREE_ISA_SSE42
                            EMBREE_ISA_AVX
                            EMBREE_ISA_AVX2
                            EMBREE_ISA_AVX512
                            EMBREE_ISA_AVX512SKX
                            EMBREE_ISPC_SUPPORT
                            #
                            #EMBREE_TUTORIALS
                            EMBREE_RAY_MASK
                            #EMBREE_STAT_COUNTERS
                            EMBREE_FILTER_FUNCTION
                            EMBREE_GEOMETRY_TRIANGLE
                            EMBREE_GEOMETRY_QUAD
                            EMBREE_GEOMETRY_CURVE
                            EMBREE_GEOMETRY_SUBDIVISION
                            EMBREE_GEOMETRY_USER
                            EMBREE_GEOMETRY_INSTANCE
                            EMBREE_GEOMETRY_INSTANCE_ARRAY
                            EMBREE_GEOMETRY_GRID
                            EMBREE_GEOMETRY_POINT
                            EMBREE_RAY_PACKETS
                            EMBREE_MAX_INSTANCE_LEVEL_COUNT=1
                            EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR=2.0
                            EMBREE_DISC_POINT_SELF_INTERSECTION_AVOIDANCE
                        EXTRA_INCLUDES
                            ${MRAY_PLATFORM_LIB_DIRECTORY}/embree/include
                        )
endif()

# Header only libraries
# Create mock targets, project layout enables direct access to the headers
# These are generated for verbosity
add_library(json INTERFACE)
target_compile_definitions(json INTERFACE JSON_DIAGNOSTICS=1)
add_library(nlohmann::json ALIAS json)

add_library(cli11 INTERFACE)
add_library(cli11::cli11 ALIAS cli11)

gen_imported_target(NAME fmt::fmt STATIC DEBUG_TARGET fmt RELEASE_TARGET fmt
                    EXTRA_DEFINES FMT_USE_NONTYPE_TEMPLATE_ARGS=0)
gen_imported_target(NAME assimp::assimp SHARED DEBUG_TARGET assimpd RELEASE_TARGET assimp VERSION 5)
gen_imported_target(NAME gfgfileio::gfgfileio STATIC DEBUG_TARGET GFGFileIOd RELEASE_TARGET GFGFileIO)
gen_imported_target(NAME oiio::oiio SHARED DEBUG_TARGET OpenImageIOd RELEASE_TARGET OpenImageIO VERSION 3.0)
gen_imported_target(NAME oiio::oiio_util SHARED DEBUG_TARGET OpenImageIO_Utild RELEASE_TARGET OpenImageIO_Util VERSION 3.0)
gen_imported_target(NAME spdlog::spdlog SHARED DEBUG_TARGET spdlogd RELEASE_TARGET spdlog VERSION 1.14
                    EXTRA_DEFINES SPDLOG_FMT_EXTERNAL SPDLOG_COMPILED_LIB)
gen_imported_target(NAME imgui::imgui STATIC DEBUG_TARGET imguid RELEASE_TARGET imgui)

# Different names on different platforms
if(WIN32)
    gen_imported_target(NAME vulkan::vulkan SHARED DEBUG_TARGET vulkan-1d RELEASE_TARGET vulkan-1)
    gen_imported_target(NAME glfw3::glfw3 SHARED DEBUG_TARGET glfw3ddll RELEASE_TARGET glfw3dll)
else()
    gen_imported_target(NAME vulkan::vulkan SHARED DEBUG_TARGET vulkand RELEASE_TARGET vulkan VERSION 1)
    gen_imported_target(NAME glfw3::glfw3 SHARED DEBUG_TARGET glfwd RELEASE_TARGET glfw VERSION 3)
endif()

if(MRAY_ENABLE_USD)
    if(WIN32)
        set(MRAY_TBB_DEBUG_NAME tbb12_debug)
        set(MRAY_TBB_RELEASE_NAME tbb12)
    else()
        set(MRAY_TBB_DEBUG_NAME tbb_debug)
        set(MRAY_TBB_RELEASE_NAME tbb)
    endif()
    # Monolithic Build of USD
    # gen_imported_target(NAME usd::usd SHARED
    #                     OVERRIDE_LIB_PREFIX ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/lib
    #                     DEBUG_TARGET        usd_msd
    #                     RELEASE_TARGET      usd_ms
    #                     EXTRA_INCLUDES      ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/include)
    # Above one (with USD_MONOLITHIC=ON "Ext" side) does not properly generate the shared
    # library on Linux. So we manually put all the libs to the MRaySceneLoaderUSD.dll
    # Achieve Build of USD
    gen_archive_pack(NAME usd::usd_archive
                     HAS_DEBUG_SUFFIX PACKAGE_WHOLE_ARCHIVE
                     EXTRA_INCLUDES      ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/include
                     EXTRA_LIB_FOLDERS   ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/lib
                     EXTRA_DEFINES       PXR_STATIC
                     # Lets go!
                     ARCHIVES
                        usd_usdLux
                        usd_usdProc
                        usd_usdMedia
                        usd_usdVol
                        usd_usdSemantics
                        usd_usdShade
                        usd_usdSkel
                        usd_usdGeom
                        usd_usd
                        usd_pcp
                        usd_sdr
                        usd_ndr
                        usd_sdf
                        usd_kind
                        usd_ar
                        usd_ts
                        usd_vt
                        usd_plug
                        usd_work
                        usd_trace
                        usd_js
                        usd_pegtl
                        usd_gf
                        usd_tf
                        usd_arch
                     )

    gen_imported_target(NAME tbb::tbb SHARED
                        DEBUG_TARGET ${MRAY_TBB_DEBUG_NAME}
                        RELEASE_TARGET ${MRAY_TBB_RELEASE_NAME}
                        VERSION 12)
endif()

# gen_imported_target(NAME imgui::imgui_ti STATIC DEBUG_TARGET imgui_tex_inspectd RELEASE_TARGET imgui_tex_inspect)

if(MRAY_BUILD_TESTS)
    gen_imported_target(NAME gtest::gtest SHARED DEBUG_TARGET gtest RELEASE_TARGET gtest VERSION 1.13.0)
    gen_imported_target(NAME gtest::gtest_main SHARED DEBUG_TARGET gtest_main RELEASE_TARGET gtest_main VERSION 1.13.0)
    gen_imported_target(NAME gtest::gmock SHARED DEBUG_TARGET gmock RELEASE_TARGET gmock VERSION 1.13.0)
    gen_imported_target(NAME gtest::gmock_main SHARED DEBUG_TARGET gmock_main RELEASE_TARGET gmock_main VERSION 1.13.0)
endif()
