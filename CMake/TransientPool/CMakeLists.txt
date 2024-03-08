set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/TransientPool)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/TransientPool.h
    ${CURRENT_SOURCE_DIR}/TransientPool.hpp
    ${CURRENT_SOURCE_DIR}/TransientPool.cpp)

source_group("" FILES ${SRC_COMMON})
add_library(TransientPool SHARED ${SRC_COMMON})

target_link_libraries(TransientPool PRIVATE
                      CoreLib
                      mray::meta_compile_opts)

target_compile_definitions(TransientPool
                           PRIVATE
                           MRAY_TRANSIENT_POOL_SHARED_EXPORT)