set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

set(SRC_SUBCOMMAND
    ${CURRENT_SOURCE_DIR}/ConvertCommand.cpp
    ${CURRENT_SOURCE_DIR}/ConvertCommand.h
    ${CURRENT_SOURCE_DIR}/CommandI.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/main.cpp)

source_group("" FILES ${SRC_COMMON})
source_group("Subcommands" FILES ${SRC_SUBCOMMAND})

set(SRC_ALL
    ${SRC_SUBCOMMAND}
    ${SRC_COMMON})

add_executable(MRay ${SRC_ALL})

target_link_libraries(MRay PRIVATE
                      CoreLib
                      GFGConverter
                      cli11::cli11
                      mray::meta_compile_opts)

# Not-linked but dependent targets
add_dependencies(MRay
                 TracerCUDA)

# Debug Parameters (For my own use case)
if(MSVC)

    set(DEBUG_ARGS
        "-t TracerConfig.json -v VisorConfig.json -r 640x360 Scenes/modernPool.json")

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()