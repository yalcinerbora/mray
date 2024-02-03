set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRayInput)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/MRayInput.h
    ${CURRENT_SOURCE_DIR}/MRayInput.hpp
    ${CURRENT_SOURCE_DIR}/MRayInput.cpp)

source_group("" FILES ${SRC_COMMON})
add_library(MRayInput SHARED ${SRC_COMMON})

target_link_libraries(MRayInput PRIVATE
                      CoreLib
                      mray::meta_compile_opts)

target_compile_definitions(MRayInput
                           PRIVATE
                           MRAY_INPUT_SHARED_EXPORT)