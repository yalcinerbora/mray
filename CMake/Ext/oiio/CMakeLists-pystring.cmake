# This code is

# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

cmake_minimum_required(VERSION 3.21)
include(GNUInstallDirs)

project(pystring VERSION 1.1.3 LANGUAGES CXX)

add_library(pystring STATIC)
target_sources(pystring PUBLIC FILE_SET HEADERS
               BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
               FILES
               ${CMAKE_CURRENT_SOURCE_DIR}/pystring.h)
set_property(TARGET pystring PROPERTY PUBLIC_HEADER
             ${CMAKE_CURRENT_SOURCE_DIR}/pystring.h)

target_sources(pystring PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/pystring.cpp)

set_target_properties(pystring PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)


install(TARGETS pystring
        EXPORT pystringTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pystring)

install(TARGETS pystring PUBLIC_HEADER
        DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/pystring)

# From the cmake docs tutorial
include(CMakePackageConfigHelpers)

set(PYSTRING_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/pystring)
export(TARGETS pystring NAMESPACE pystring::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/pystringTargets.cmake")

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/pystringConfig.cmake.in
     " @PACKAGE_INIT@
     include (\"\${CMAKE_CURRENT_LIST_DIR}/pystringTargets.cmake\")")

# generate the config file that includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/pystringConfig.cmake.in
                             "${CMAKE_CURRENT_BINARY_DIR}/pystringConfig.cmake"
                             INSTALL_DESTINATION ${PYSTRING_CONFIG_DIR}
                             NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pystringConfigVersion.cmake"
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/pystringConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/pystringConfigVersion.cmake"
        DESTINATION ${PYSTRING_CONFIG_DIR})

install(EXPORT pystringTargets
        NAMESPACE pystring::
        FILE pystringTargets.cmake
        DESTINATION ${PYSTRING_CONFIG_DIR})
