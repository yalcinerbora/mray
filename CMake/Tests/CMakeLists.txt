include(GoogleTest)

set(MRAY_TEST_TARGETS_FOLDER "Tests")

# TODO: Selectively enable/disable device tests
add_subdirectory(TDeviceCUDA)
add_subdirectory(TTracerCUDA)
