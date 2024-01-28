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

# Debug Parameters (For my own use case)
if(MSVC)

    set(DEBUG_ARGS
        "-t TracerConfig.json -v VisorConfig.json -r 640x360 Scenes/modernPool.json")

    set_target_properties(MRayInput PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()