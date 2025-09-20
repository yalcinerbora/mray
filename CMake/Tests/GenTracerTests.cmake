function(gen_tracer_test)
    # Parse Args
    set(options)
    set(oneValueArgs NAME BACKEND)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_TRACER_TEST "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(CURRENT_SOURCE_DIR ${MRAY_TEST_SOURCE_DIRECTORY}/Tracer)

    set(SRC_COMMON
        ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
        ${CURRENT_SOURCE_DIR}/T_Random.cu
        ${CURRENT_SOURCE_DIR}/T_Spectrum.cu
        ${CURRENT_SOURCE_DIR}/T_Filters.cu
        ${CURRENT_SOURCE_DIR}/T_RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/T_DefaultTriangle.cu
        ${CURRENT_SOURCE_DIR}/T_DefaultLights.cu
        ${CURRENT_SOURCE_DIR}/T_Distributions.cu
        ${CURRENT_SOURCE_DIR}/T_Materials.cu
        ${CURRENT_SOURCE_DIR}/T_StreamingTexture.cu
        ${CURRENT_SOURCE_DIR}/T_RayCone.cu)

    source_group("" FILES ${SRC_COMMON})

    # I dunno why but global set did not work
    change_device_source_file_language(
        MACRO
        ${GEN_TRACER_TEST_BACKEND}
        SOURCE_FILES
        ${CURRENT_SOURCE_DIR}/T_Random.cu
        ${CURRENT_SOURCE_DIR}/T_Spectrum.cu
        ${CURRENT_SOURCE_DIR}/T_Filters.cu
        ${CURRENT_SOURCE_DIR}/T_RayPartitioner.cu
        ${CURRENT_SOURCE_DIR}/T_DefaultTriangle.cu
        ${CURRENT_SOURCE_DIR}/T_DefaultLights.cu
        ${CURRENT_SOURCE_DIR}/T_Distributions.cu
        ${CURRENT_SOURCE_DIR}/T_Materials.cu
        ${CURRENT_SOURCE_DIR}/T_StreamingTexture.cu
        ${CURRENT_SOURCE_DIR}/T_RayCone.cu
        ${CURRENT_SOURCE_DIR}/T_Color.cu
    )

    set(TARGET_FULL_NAME "TTracer${GEN_TRACER_TEST_NAME}")
    set(TRACER_TARGET_FULL_NAME "Tracer${GEN_TRACER_TEST_NAME}")

    add_executable(${TARGET_FULL_NAME} ${SRC_COMMON})
    set_target_properties(${TARGET_FULL_NAME} PROPERTIES OUTPUT_NAME "T_Tracer${GEN_TRACER_TEST_NAME}")

    target_link_libraries(${TARGET_FULL_NAME} PRIVATE
                          ${TRACER_TARGET_FULL_NAME}
                          mray::meta_compile_opts
                          mray::test_common
                          gtest::gtest
                          gtest::gtest_main)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON)

    if(GEN_TRACER_TEST_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")
        target_link_libraries(${TARGET_FULL_NAME} PRIVATE
                              mray::cuda_extra_compile_opts)

        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    elseif(GEN_TRACER_TEST_BACKEND STREQUAL "MRAY_GPU_BACKEND_CPU")
        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              INTERPROCEDURAL_OPTIMIZATION_RELEASE
                              ${MRAY_HOST_BACKEND_IPO_MODE})
    endif()

    add_precompiled_headers(TARGET ${TARGET_FULL_NAME})

    if(WIN32)
        set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")
        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                              VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
    endif()

    add_custom_command(TARGET ${TARGET_FULL_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE:gtest::gtest>"
                       "$<TARGET_FILE:gtest::gtest_main>"
                       "$<TARGET_FILE:gtest::gmock>"
                       ${MRAY_CONFIG_BIN_DIRECTORY})

    set(GEN_TRACER_TEST_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)

endfunction()
