set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/SceneLoaderUSD)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/SceneLoaderUSD.h
    ${CURRENT_SOURCE_DIR}/SceneLoaderUSD.cpp
    ${CURRENT_SOURCE_DIR}/MeshProcessor.h
    ${CURRENT_SOURCE_DIR}/MeshProcessor.cpp
    ${CURRENT_SOURCE_DIR}/MaterialProcessor.h
    ${CURRENT_SOURCE_DIR}/MaterialProcessor.cpp
    ${CURRENT_SOURCE_DIR}/MRayUSDTypes.h)

source_group("" FILES ${SRC_COMMON})
add_library(SceneLoaderUSD SHARED ${SRC_COMMON})

target_include_directories(SceneLoaderUSD
                           PRIVATE
                           ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/include)

target_link_libraries(SceneLoaderUSD
                      PRIVATE
                      CoreLib
                      ImageLoader
                      TransientPool
                      mray::meta_compile_opts
                      # USD-related
                      usd::usd_archive
                      tbb::tbb)

if(WIN32)
    target_link_libraries(SceneLoaderUSD PRIVATE
                          Ws2_32.lib
                          Shlwapi.lib
                          Dbghelp.lib
                         )
endif()

target_compile_definitions(SceneLoaderUSD
                           PRIVATE
                           MRAY_SCENELOADER_USD_SHARED_EXPORT)

add_precompiled_headers(TARGET SceneLoaderUSD)

# Copy DLLS
if(WIN32)
    add_custom_command(TARGET SceneLoaderUSD POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/tbb12$<$<CONFIG:Debug>:_debug>.dll"
                       ${MRAY_CONFIG_BIN_DIRECTORY}
                       COMMENT "Copying USD DLLs")
else()
    add_custom_command(TARGET SceneLoaderUSD POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE:tbb::tbb>"
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()

# USD Plugin folder
add_custom_command(TARGET SceneLoaderUSD POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                   ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/lib/usd
                   ${MRAY_CONFIG_BIN_DIRECTORY}/usd
                   COMMENT "Copying USD plugins")