set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/SceneLoaderMRay)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/SceneLoaderMRay.h
    ${CURRENT_SOURCE_DIR}/SceneLoaderMRay.cpp
    ${CURRENT_SOURCE_DIR}/MeshLoaderMRayJson.h
    ${CURRENT_SOURCE_DIR}/MeshLoaderMRayJson.cpp
    ${CURRENT_SOURCE_DIR}/JsonNode.h
    ${CURRENT_SOURCE_DIR}/JsonNode.hpp
    ${CURRENT_SOURCE_DIR}/NodeNames.h)

source_group("" FILES ${SRC_COMMON})
add_library(SceneLoaderMRay SHARED ${SRC_COMMON})

target_link_libraries(SceneLoaderMRay
                      PRIVATE
                      CoreLib
                      MRayInput
                      mray::meta_compile_opts
                      nlohmann::json)

target_compile_definitions(SceneLoaderMRay
                           PRIVATE
                           MRAY_SCENELOADER_MRAY_SHARED_EXPORT)