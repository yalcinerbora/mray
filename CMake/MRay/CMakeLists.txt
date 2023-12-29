set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

set(SRC_COMMON ${CURRENT_SOURCE_DIR}/main.cpp)

source_group("" FILES ${SRC_COMMON})
add_executable(MRay ${SRC_COMMON})

target_link_libraries(MRay PRIVATE
                      CoreLib
                      mray::meta_compile_opts)

# Not-linked but dependent targets
# add_dependencies(MRay
#                  TracerCUDA
#                  VisorGL)

# Debug Parameters (For my own use case)
if(MSVC)

    set(DEBUG_ARGS
        "-t TracerConfig.json -v VisorConfig.json -r 640x360 Scenes/modernPool.json")

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()