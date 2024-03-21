set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tests/Core)

message(STATUS "Configuring ${CURRENT_SOURCE_DIR}")

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/T_GraphicsFunctions.cpp
    ${CURRENT_SOURCE_DIR}/T_MPMCQueue.cpp
)

source_group("" FILES ${SRC_COMMON})

add_executable(TCore ${SRC_COMMON})

target_link_libraries(TCore PRIVATE
                      CoreLib
                      mray::meta_compile_opts
                      mray::test_common
                      gtest::gtest
                      gtest::gtest_main

                      # This is requires on linux since currently tests do not have
                      # and .cu files so CMake does not add CUDA related libs
                      CUDA::cudart
                      )

gtest_discover_tests(TCore)

set_target_properties(TCore PROPERTIES FOLDER ${MRAY_TEST_TARGETS_FOLDER})
set_target_properties(TCore PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(MSVC)
    add_custom_command(TARGET TCore POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gtest_main.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gmock_main.dll"
                        ${MRAY_CONFIG_BIN_DIRECTORY})
    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")
    set_target_properties(TCore PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()