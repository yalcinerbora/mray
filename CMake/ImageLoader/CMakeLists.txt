set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/ImageLoader)

# SOURCES
set(SRC_ALL
    ${CURRENT_SOURCE_DIR}/ImageLoaderI.h
    ${CURRENT_SOURCE_DIR}/ImageLoader.h
    ${CURRENT_SOURCE_DIR}/ImageLoader.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp)

source_group("" FILES ${SRC_ALL})

# DLL
add_library(ImageLoader SHARED ${SRC_ALL})

target_compile_definitions(ImageLoader PRIVATE
                           MRAY_IMAGELOADER_SHARED_EXPORT)

target_link_libraries(ImageLoader PRIVATE
                      CoreLib
                      TransientPool
                      mray::meta_compile_opts
                      oiio::oiio
                      oiio::oiio_util)

# Copy DLLS
if(MSVC)
    add_custom_command(TARGET ImageLoader POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/OpenImageIO$<$<CONFIG:Debug>:d>.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/OpenImageIO_Util$<$<CONFIG:Debug>:d>.dll"
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()