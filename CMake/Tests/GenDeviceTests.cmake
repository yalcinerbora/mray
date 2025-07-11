function(gen_device_test)
    # Parse Args
    set(options)
    set(oneValueArgs NAME BACKEND)
    set(multiValueArgs)

    cmake_parse_arguments(GEN_DEVICE_TEST "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(CURRENT_SOURCE_DIR ${MRAY_TEST_SOURCE_DIRECTORY}/Device)

    set(SRC_COMMON
    ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
    ${CURRENT_SOURCE_DIR}/T_DeviceMemory.cpp
    ${CURRENT_SOURCE_DIR}/T_TextureTypes.h
    ${CURRENT_SOURCE_DIR}/T_Texture.cu
    ${CURRENT_SOURCE_DIR}/T_KernelCall.cu
    ${CURRENT_SOURCE_DIR}/T_AlgTypes.h
    ${CURRENT_SOURCE_DIR}/T_AlgScan.cu
    ${CURRENT_SOURCE_DIR}/T_AlgReduce.cu
    ${CURRENT_SOURCE_DIR}/T_AlgRadixSort.cu
    ${CURRENT_SOURCE_DIR}/T_AlgBinPartition.cu
    ${CURRENT_SOURCE_DIR}/T_AlgCommon.cu)

    source_group("" FILES ${SRC_COMMON})

    # I dunno why but global set did not work
    change_device_source_file_language(
        MACRO
        ${GEN_DEVICE_TEST_BACKEND}
        SOURCE_FILES
        ${CURRENT_SOURCE_DIR}/T_Texture.cu
        ${CURRENT_SOURCE_DIR}/T_KernelCall.cu
        ${CURRENT_SOURCE_DIR}/T_AlgScan.cu
        ${CURRENT_SOURCE_DIR}/T_AlgReduce.cu
        ${CURRENT_SOURCE_DIR}/T_AlgRadixSort.cu
        ${CURRENT_SOURCE_DIR}/T_AlgBinPartition.cu
        ${CURRENT_SOURCE_DIR}/T_AlgCommon.cu
    )

    set(TARGET_FULL_NAME "TDevice${GEN_DEVICE_TEST_NAME}")
    set(DEVICE_TARGET_FULL_NAME "Device${GEN_DEVICE_TEST_NAME}")
    add_executable(${TARGET_FULL_NAME} ${SRC_COMMON})

    target_link_libraries(${TARGET_FULL_NAME} PRIVATE
                          ${DEVICE_TARGET_FULL_NAME}
                          mray::meta_compile_opts
                          mray::test_common
                          gtest::gtest
                          gtest::gtest_main)

    set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                          POSITION_INDEPENDENT_CODE ON)

    if(GEN_DEVICE_TEST_BACKEND STREQUAL "MRAY_GPU_BACKEND_CUDA")
        target_link_libraries(${TARGET_FULL_NAME} PRIVATE
                              mray::cuda_extra_compile_opts)

        set_target_properties(${TARGET_FULL_NAME} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
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


    set(GEN_DEVICE_TEST_TARGET_NAME ${TARGET_FULL_NAME} PARENT_SCOPE)

endfunction()
