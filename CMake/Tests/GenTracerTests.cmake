function(gen_tracer_test)
    # Parse Args
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_TRACER_TEST "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(CURRENT_SOURCE_DIR ${MRAY_TEST_SOURCE_DIRECTORY}/Tracer)

    set(SRC_COMMON
        ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
        ${CURRENT_SOURCE_DIR}/T_Random.cu
        ${CURRENT_SOURCE_DIR}/T_Filters.cu
        ${CURRENT_SOURCE_DIR}/T_RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/T_DefaultTriangle.cu
        ${CURRENT_SOURCE_DIR}/T_DefaultLights.cu
        ${CURRENT_SOURCE_DIR}/T_Distributions.cu
        ${CURRENT_SOURCE_DIR}/T_Materials.cu)

    source_group("" FILES ${SRC_COMMON})

    set(TARGET_FULL_NAME "TTracer${GEN_TRACER_TEST_NAME}")
    set(TRACER_TARGET_FULL_NAME "Tracer${GEN_TRACER_TEST_NAME}")
    add_executable(${TARGET_FULL_NAME} ${SRC_COMMON})

    target_link_libraries(${TARGET_FULL_NAME} PRIVATE
                          ${TRACER_TARGET_FULL_NAME}
                          mray::meta_compile_opts
                          mray::cuda_extra_compile_opts
                          mray::test_common
                          gtest::gtest
                          gtest::gtest_main)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION ON
                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})

    if(MSVC)
        add_custom_command(TARGET ${TARGET_FULL_NAME} PRE_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gtest_main.dll"
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                           "${MRAY_CONFIG_LIB_DIRECTORY}/gmock_main.dll"
                            ${MRAY_CONFIG_BIN_DIRECTORY})
        set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")
        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                              VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
    endif()

    set(GEN_TRACER_TEST_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)

endfunction()
