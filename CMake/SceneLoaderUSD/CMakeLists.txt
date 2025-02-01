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

target_link_libraries(SceneLoaderUSD
                      PRIVATE
                      CoreLib
                      ImageLoader
                      TransientPool
                      mray::meta_compile_opts
                      usd::usd
                      # Why do we need this? usd is a dll?
                      tbb::tbb)

target_compile_definitions(SceneLoaderUSD
                           PRIVATE
                           MRAY_SCENELOADER_USD_SHARED_EXPORT)

add_precompiled_headers(TARGET SceneLoaderUSD)

# Copy DLLS
if(WIN32)
    add_custom_command(TARGET SceneLoaderUSD POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/tbb12$<$<CONFIG:Debug>:_debug>.dll"
                       "${MRAY_PLATFORM_LIB_DIRECTORY}/usd/lib/usd_ms$<$<CONFIG:Debug>:d>.dll"
                       ${MRAY_CONFIG_BIN_DIRECTORY}
                       COMMENT "Copying USD DLLs")
    # USD Plugin folder
    add_custom_command(TARGET SceneLoaderUSD POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                       ${MRAY_PLATFORM_LIB_DIRECTORY}/usd/lib/usd
                       ${MRAY_CONFIG_BIN_DIRECTORY}/usd
                       COMMENT "Copying USD plugins")
endif()