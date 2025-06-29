include("GenDeviceTests")
include("GenTracerTests")

# Target Generation
gen_device_test(NAME "CPU" BACKEND "MRAY_GPU_BACKEND_CPU")
gen_tracer_test(NAME "CPU" BACKEND "MRAY_GPU_BACKEND_CPU")

gtest_discover_tests(${GEN_DEVICE_TEST_TARGET_NAME})
gtest_discover_tests(${GEN_TRACER_TEST_TARGET_NAME})

set_target_properties(${GEN_DEVICE_TEST_TARGET_NAME} PROPERTIES
                      FOLDER "${MRAY_TEST_TARGETS_FOLDER}/CPU")
set_target_properties(${GEN_TRACER_TEST_TARGET_NAME} PROPERTIES
                      FOLDER "${MRAY_TEST_TARGETS_FOLDER}/CPU")