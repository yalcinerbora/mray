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
set(SRC_COMMON_GLOBAL
    ${MRAY_SOURCE_DIRECTORY}/Common/RenderImageStructs.h
    ${MRAY_SOURCE_DIRECTORY}/Common/AnalyticStructs.h
    ${MRAY_SOURCE_DIRECTORY}/Common/TransferQueue.h
    ${MRAY_SOURCE_DIRECTORY}/Common/JsonCommon.h)

source_group("Tracer" FILES ${SRC_TRACER})
source_group("Subcommands" FILES ${SRC_SUBCOMMAND})
source_group("Common" FILES ${SRC_COMMON_GLOBAL})
source_group("" FILES ${SRC_COMMON})

set(SRC_ALL
    ${SRC_TRACER}
    ${SRC_SUBCOMMAND}
    ${SRC_COMMON}
    ${SRC_COMMON_GLOBAL})

add_executable(MRay ${SRC_ALL})

target_link_libraries(MRay PRIVATE
                      CoreLib
                      GFGConverter
                      TransientPool
                      ImageLoader
                      cli11::cli11
                      bs::thread_pool
                      mray::meta_compile_opts)

add_precompiled_headers(TARGET MRay)

# Not-linked but dependent targets
# Dynamically loaded DLLs
add_dependencies(MRay SceneLoaderMRay)

# Same as above, but USD may not be built
if(MRAY_ENABLE_USD)
    add_dependencies(MRay SceneLoaderUSD)
endif()

# Tracer Related dependencies
if(${MRAY_DEVICE_BACKEND} STREQUAL "MRAY_GPU_BACKEND_CUDA")
    add_dependencies(MRay TracerDLL_CUDA)
elseif(${MRAY_DEVICE_BACKEND} STREQUAL "MRAY_GPU_BACKEND_HIP")
    add_dependencies(MRay TracerDLL_HIP)
endif()

if(MRAY_BUILD_VISOR)
    add_dependencies(MRay Visor)
endif()

# Debug Parameters (For my own use case)
if(MSVC)
    set(DEBUG_ARGS "")
    if(MRAY_BUILD_VISOR)
        string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "visor ")
    else()
        string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "run ")
    endif()
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "-r 640x360 ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "-s Scenes/helloTriangle.json ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "--tConf tracerConfig.json ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "--vConf visorConfig.json ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "--rConf renderConfig.json ")

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()