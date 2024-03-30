set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/GFGConverter)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/GFGConverter.h
    ${CURRENT_SOURCE_DIR}/GFGConverter.cpp)

source_group("" FILES ${SRC_COMMON})
add_library(GFGConverter SHARED ${SRC_COMMON})

target_link_libraries(GFGConverter
                      PRIVATE
                      CoreLib
                      mray::meta_compile_opts
                      nlohmann::json
                      assimp::assimp
                      gfgfileio::gfgfileio)

target_compile_definitions(GFGConverter
                           PRIVATE
                           MRAY_GFGCONVERTER_SHARED_EXPORT)