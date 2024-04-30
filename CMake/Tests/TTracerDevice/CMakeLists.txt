include("GenDeviceTests")
include("GenTracerTests")

# Target Generation
gen_device_test(NAME ${MRAY_GPU_PLATFORM_NAME})
gen_tracer_test(NAME ${MRAY_GPU_PLATFORM_NAME})

gtest_discover_tests(${GEN_DEVICE_TEST_TARGET_NAME})
gtest_discover_tests(${GEN_TRACER_TEST_TARGET_NAME})

set_target_properties(${GEN_DEVICE_TEST_TARGET_NAME} PROPERTIES
                      FOLDER "${MRAY_TEST_TARGETS_FOLDER}/${MRAY_GPU_PLATFORM_NAME}")
set_target_properties(${GEN_TRACER_TEST_TARGET_NAME} PROPERTIES
                      FOLDER "${MRAY_TEST_TARGETS_FOLDER}/${MRAY_GPU_PLATFORM_NAME}")