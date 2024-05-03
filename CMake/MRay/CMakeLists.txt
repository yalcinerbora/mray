set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

set(SRC_SUBCOMMAND
    ${CURRENT_SOURCE_DIR}/ConvertCommand.cpp
    ${CURRENT_SOURCE_DIR}/ConvertCommand.h
    ${CURRENT_SOURCE_DIR}/VisorCommand.cpp
    ${CURRENT_SOURCE_DIR}/VisorCommand.h
    ${CURRENT_SOURCE_DIR}/RunCommand.cpp
    ${CURRENT_SOURCE_DIR}/RunCommand.h
    ${CURRENT_SOURCE_DIR}/CommandI.h)

set(SRC_TRACER
    ${CURRENT_SOURCE_DIR}/TracerThread.h
    ${CURRENT_SOURCE_DIR}/TracerThread.cpp)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/main.cpp)

# TODO: Move this somwhere better?
# (TracerLib/DLL, Visor.dll, MRay.exe both use these)
# But we need to show this on VS Solution Explorer so adding here
set(SRC_COMMON_HEADERS
    ${MRAY_SOURCE_DIRECTORY}/CommonHeaders/RenderImageStructs.h
    ${MRAY_SOURCE_DIRECTORY}/CommonHeaders/AnalyticStructs.h
    ${MRAY_SOURCE_DIRECTORY}/CommonHeaders/TransferQueue.h)

source_group("Tracer" FILES ${SRC_TRACER})
source_group("Subcommands" FILES ${SRC_SUBCOMMAND})
source_group("Common" FILES ${SRC_COMMON_HEADERS})
source_group("" FILES ${SRC_COMMON})

set(SRC_ALL
    ${SRC_TRACER}
    ${SRC_SUBCOMMAND}
    ${SRC_COMMON}
    ${SRC_COMMON_HEADERS})

add_executable(MRay ${SRC_ALL})

target_link_libraries(MRay PRIVATE
                      CoreLib
                      GFGConverter
                      TransientPool
                      cli11::cli11
                      mray::meta_compile_opts)

# Not-linked but dependent targets
add_dependencies(MRay
                 Visor)

# Debug Parameters (For my own use case)
if(MSVC)

    set(DEBUG_ARGS "")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "-r 640x360 ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "-s Scenes/texture.json ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "--tConf TracerConfig.json ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "--vConf VisorConfig.json ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "--rConf RenderConfig.json ")

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()