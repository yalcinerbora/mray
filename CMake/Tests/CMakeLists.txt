include(GoogleTest)

set(MRAY_TEST_TARGETS_FOLDER "Tests")

add_library(test_common INTERFACE)
target_include_directories(test_common INTERFACE ${MRAY_SOURCE_DIRECTORY}/Tests)

add_library(mray::test_common ALIAS test_common)

# TODO: Selectively enable/disable device tests
add_subdirectory(TCore)
add_subdirectory(TDeviceCUDA)
add_subdirectory(TTracerCUDA)
add_subdirectory(TSceneLoaderMRay)
