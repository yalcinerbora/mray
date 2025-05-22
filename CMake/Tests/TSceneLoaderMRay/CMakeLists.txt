set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Tests/SceneLoaderMRay)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/T_SceneLoaderMRay.cpp
    ${CURRENT_SOURCE_DIR}/MockTracer.h
    ${CURRENT_SOURCE_DIR}/TestScenes.h
)

source_group("" FILES ${SRC_COMMON})

add_executable(TSceneLoaderMRay ${SRC_COMMON})

target_link_libraries(TSceneLoaderMRay
                      PRIVATE
                      CoreLib
                      TransientPool
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts
                      mray::test_common
                      gtest::gtest
                      gtest::gtest_main)

add_dependencies(TSceneLoaderMRay SceneLoaderMRay MeshLoader)

gtest_discover_tests(TSceneLoaderMRay)

set_target_properties(TSceneLoaderMRay PROPERTIES FOLDER ${MRAY_TEST_TARGETS_FOLDER})
set_target_properties(TSceneLoaderMRay PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

add_precompiled_headers(TARGET TSceneLoaderMRay)

if(WIN32)
    add_custom_command(TARGET TSceneLoaderMRay PRE_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gtest_main.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                        "${MRAY_CONFIG_LIB_DIRECTORY}/gmock_main.dll"
                        ${MRAY_CONFIG_BIN_DIRECTORY})
    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")
    set_target_properties(TSceneLoaderMRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()