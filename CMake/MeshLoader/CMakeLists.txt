set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MeshLoader)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/MeshLoaderGFG.h
    ${CURRENT_SOURCE_DIR}/MeshLoaderGFG.cpp
    ${CURRENT_SOURCE_DIR}/MeshLoaderAssimp.h
    ${CURRENT_SOURCE_DIR}/MeshLoaderAssimp.cpp
)

source_group("" FILES ${SRC_COMMON})
add_library(MeshLoader SHARED ${SRC_COMMON})

target_link_libraries(MeshLoader
                      PRIVATE
                      CoreLib
                      TransientPool
                      mray::meta_compile_opts
                      assimp::assimp
                      gfgfileio::gfgfileio)

target_compile_definitions(MeshLoader
                           PRIVATE
                           MRAY_MESHLOADER_SHARED_EXPORT)