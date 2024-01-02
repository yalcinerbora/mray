set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tests/DeviceCUDA)

message(STATUS "Configuring ${CURRENT_SOURCE_DIR}")

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/T_DeviceMemory.cpp)

message(STATUS ${SRC_COMMON})

source_group("" FILES ${SRC_COMMON})

add_executable(TDeviceCUDA ${SRC_COMMON})

target_link_libraries(TDeviceCUDA PRIVATE
                      DeviceLib
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts
                      gtest::gtest
                      gtest::gtest_main)

gtest_discover_tests(TDeviceCUDA)

set_target_properties(TDeviceCUDA PROPERTIES FOLDER ${MRAY_TEST_TARGETS_FOLDER})

if(MSVC)

    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")
    set_target_properties(TDeviceCUDA PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()