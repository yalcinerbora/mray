
set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/TracerKernelGen)

set(SRC_ALL
    ${CURRENT_SOURCE_DIR}/main.cpp
    ${CURRENT_SOURCE_DIR}/Templates.h)

source_group("" FILES ${SRC_ALL})

add_executable(TracerKernelGen ${SRC_ALL})
target_link_libraries(TracerKernelGen
                      PRIVATE
                      CoreLib
                      mray::meta_compile_opts)

# Debug Parameters (For my own use case)
if(MSVC)
    set(DEBUG_ARGS "")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "../Source/TracerDLL/TracerTypeGenInput.txt ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "0 ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "1 ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} ". ")
    string(CONCAT DEBUG_ARGS ${DEBUG_ARGS} "MRAY_GPU_BACKEND_CUDA")

    set_target_properties(TracerKernelGen PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()

set_target_properties(TracerKernelGen PROPERTIES FOLDER ${MRAY_PREDEFINED_TARGETS_FOLDER})