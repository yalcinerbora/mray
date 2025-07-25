set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Core)

# SOURCES
set(SRC_CONTAINERS
    ${CURRENT_SOURCE_DIR}/DataStructures.h
    ${CURRENT_SOURCE_DIR}/DataStructures.hpp)

if(MSVC)
    set(SRC_CONTAINERS ${SRC_CONTAINERS}
        ${CURRENT_SOURCE_DIR}/DataStructs.natvis)
endif()

set(SRC_SYSTEM
    ${CURRENT_SOURCE_DIR}/System.h
    ${CURRENT_SOURCE_DIR}/System.cpp
    ${CURRENT_SOURCE_DIR}/ThreadPool.h
    ${CURRENT_SOURCE_DIR}/ThreadPool.cpp
    ${CURRENT_SOURCE_DIR}/SharedLibrary.h
    ${CURRENT_SOURCE_DIR}/SharedLibrary.cpp
    ${CURRENT_SOURCE_DIR}/MRayDescriptions.h
    ${CURRENT_SOURCE_DIR}/RealtimeThread.h)

set(SRC_INTERFACES
    ${CURRENT_SOURCE_DIR}/SceneLoaderI.h
    ${CURRENT_SOURCE_DIR}/TracerI.h)

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
    ${CURRENT_SOURCE_DIR}/Math.h
    ${CURRENT_SOURCE_DIR}/MathConstants.h
)

set(SRC_UTILITY
    ${CURRENT_SOURCE_DIR}/Algorithm.h
    ${CURRENT_SOURCE_DIR}/ColorFunctions.h
    ${CURRENT_SOURCE_DIR}/TypeNameGenerators.h
    ${CURRENT_SOURCE_DIR}/BitFunctions.h
    ${CURRENT_SOURCE_DIR}/DeviceVisit.h
    ${CURRENT_SOURCE_DIR}/MemAlloc.h
    ${CURRENT_SOURCE_DIR}/MemAlloc.cpp
    ${CURRENT_SOURCE_DIR}/TypeFinder.h
    ${CURRENT_SOURCE_DIR}/NormTypes.h
    ${CURRENT_SOURCE_DIR}/Log.h
    ${CURRENT_SOURCE_DIR}/Log.cpp
    ${CURRENT_SOURCE_DIR}/Error.h
    ${CURRENT_SOURCE_DIR}/Error.cpp
    ${CURRENT_SOURCE_DIR}/Expected.h
    ${CURRENT_SOURCE_DIR}/Types.h
    ${CURRENT_SOURCE_DIR}/Flag.h
    ${CURRENT_SOURCE_DIR}/GraphicsFunctions.h
    ${CURRENT_SOURCE_DIR}/TypeGenFunction.h
    ${CURRENT_SOURCE_DIR}/ShapeFunctions.h
    ${CURRENT_SOURCE_DIR}/Timer.h
    ${CURRENT_SOURCE_DIR}/Timer.cpp
    ${CURRENT_SOURCE_DIR}/TimelineSemaphore.h
    ${CURRENT_SOURCE_DIR}/MPMCQueue.h
    ${CURRENT_SOURCE_DIR}/Filesystem.h
    ${CURRENT_SOURCE_DIR}/Profiling.h)

set(SRC_DATA_TYPES
    ${CURRENT_SOURCE_DIR}/NamedEnum.h
    ${CURRENT_SOURCE_DIR}/MRayDataType.h
    ${CURRENT_SOURCE_DIR}/MRayDataType.cpp)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/Definitions.h)

set(SRC_ALL
    ${SRC_SYSTEM}
    ${SRC_INTERFACES}
    ${SRC_CONTAINERS}
    ${SRC_MATH}
    ${SRC_UTILITY}
    ${SRC_DATA_TYPES}
    ${SRC_COMMON})

# IDE Filters
source_group("System" FILES ${SRC_SYSTEM})
source_group("Interfaces" FILES ${SRC_INTERFACES})
source_group("Containers" FILES ${SRC_CONTAINERS})
source_group("Math" FILES ${SRC_MATH})
source_group("Utility" FILES ${SRC_UTILITY})
source_group("DataTypes" FILES ${SRC_DATA_TYPES})
source_group("" FILES ${SRC_COMMON})

# DLL File
add_library(CoreLib STATIC ${SRC_ALL})

target_link_libraries(CoreLib
                      PUBLIC
                      fmt::fmt
                      $<$<BOOL:${MRAY_ENABLE_TRACY}>:tracy::tracy_client>
                      PRIVATE
                      mray::meta_compile_opts)

set_target_properties(CoreLib PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

if(MRAY_ENABLE_TRACY)
    add_custom_command(TARGET CoreLib POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE:tracy::tracy_client>"
                       ${MRAY_CONFIG_BIN_DIRECTORY}
                       COMMENT "Copying TracyClient dll")
endif()

# TODO: This is somewhat a hack?
# When HIP is enabled as a language AMD does not give
# lib and include variables ?????
if(MRAY_DEVICE_BACKEND STREQUAL "MRAY_GPU_BACKEND_HIP")
    target_include_directories(CoreLib
                               PUBLIC
                               ${CMAKE_HIP_COMPILER_ROCM_ROOT}/include)
endif()

add_precompiled_headers(TARGET CoreLib)