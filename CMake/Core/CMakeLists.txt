set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Core)

# SOURCES
set(SRC_CONTAINERS
    ${CURRENT_SOURCE_DIR}/DataStructures.h)

set(SRC_MATH
    ${CURRENT_SOURCE_DIR}/AABB.h
    ${CURRENT_SOURCE_DIR}/AABB.hpp
    ${CURRENT_SOURCE_DIR}/Matrix.h
    ${CURRENT_SOURCE_DIR}/Matrix.hpp
    ${CURRENT_SOURCE_DIR}/Quaternion.h
    ${CURRENT_SOURCE_DIR}/Quaternion.hpp
    ${CURRENT_SOURCE_DIR}/Ray.h
    ${CURRENT_SOURCE_DIR}/Ray.hpp
    ${CURRENT_SOURCE_DIR}/Vector.h
    ${CURRENT_SOURCE_DIR}/Vector.hpp
    ${CURRENT_SOURCE_DIR}/MathForward.h

    ${CURRENT_SOURCE_DIR}/MathFunctions.h
    ${CURRENT_SOURCE_DIR}/GraphicsFunctions.h
    ${CURRENT_SOURCE_DIR}/ShapeFunctions.h)

set(SRC_UTILITY
    ${CURRENT_SOURCE_DIR}/TypeFinder.h
    ${CURRENT_SOURCE_DIR}/NormTypes.h
    ${CURRENT_SOURCE_DIR}/Types.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/Definitions.h)

set(SRC_ALL
    ${SRC_CONTAINERS}
    ${SRC_MATH}
    ${SRC_UTILITY}
    ${SRC_COMMON})

# IDE Filters
source_group("Containers" FILES ${SRC_CONTAINERS})
source_group("Math" FILES ${SRC_MATH})
source_group("Utility" FILES ${SRC_UTILITY})
source_group("" FILES ${SRC_COMMON})

# DLL File
add_library(CoreLib INTERFACE ${SRC_ALL})

target_link_libraries(CoreLib
                      INTERFACE fmt::fmt
                      INTERFACE mray::meta_compile_opts)

# set_target_properties(CoreLib PROPERTIES
#                       POSITION_INDEPENDENT_CODE ON
#                       CUDA_SEPARABLE_COMPILATION ON
#                       CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Copy DLLS
# if(MSVC)
#     add_custom_command(TARGET CoreLib POST_BUILD
#                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                        "${MRAY_CONFIG_LIB_DIRECTORY}/spdlog$<$<CONFIG:Debug>:d>.dll"
#                        "${MRAY_CONFIG_LIB_DIRECTORY}/ZLIB$<$<CONFIG:Debug>:d>1.dll"
#                        ${MRAY_CONFIG_BIN_DIRECTORY})
# endif()