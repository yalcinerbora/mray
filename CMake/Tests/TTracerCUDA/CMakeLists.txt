set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tests/TracerCUDA)

message(STATUS "Configuring ${CURRENT_SOURCE_DIR}")

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/T_RayPartitioner.cu
    ${CURRENT_SOURCE_DIR}/T_DefaultTriangle.cu
    ${CURRENT_SOURCE_DIR}/T_DefaultLights.cu
    ${CURRENT_SOURCE_DIR}/T_Distributions.cu
)

source_group("" FILES ${SRC_COMMON})

add_executable(TTracerCUDA ${SRC_COMMON})

target_link_libraries(TTracerCUDA PRIVATE
                      TracerCUDA_Objects
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts
                      mray::test_common
                      gtest::gtest
                      gtest::gtest_main)

gtest_discover_tests(TTracerCUDA)

set_target_properties(TTracerCUDA PROPERTIES FOLDER ${MRAY_TEST_TARGETS_FOLDER})
set_target_properties(TTracerCUDA PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(MSVC)
    add_custom_command(TARGET TTracerCUDA PRE_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gtest_main.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gmock_main.dll"
                        ${MRAY_CONFIG_BIN_DIRECTORY})
    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")
    set_target_properties(TTracerCUDA PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()