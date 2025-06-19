include(GoogleTest)

set(MRAY_TEST_TARGETS_FOLDER "Tests")
set(MRAY_TEST_SOURCE_DIRECTORY "${MRAY_SOURCE_DIRECTORY}/Tests")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(test_common INTERFACE)
target_include_directories(test_common INTERFACE ${MRAY_SOURCE_DIRECTORY}/Tests)

add_library(mray::test_common ALIAS test_common)

# TODO: Selectively enable/disable device tests
add_subdirectory(TCore)
add_subdirectory(TSceneLoaderMRay)

if(NOT MRAY_DISABLE_DEVICE_BACKEND)
    add_subdirectory(TTracerCUDA)
endif()

if(MRAY_BUILD_HOST_BACKEND)
    add_subdirectory(TTracerCPU)
endif()

