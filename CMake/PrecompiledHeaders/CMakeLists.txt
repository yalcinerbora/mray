
if(NOT MRAY_ENABLE_PCH)
    return()
else()
    return()
endif()

# TODO: Global version does not work currently
# we need to rcheck this later
set(SRC_HEADERS
    ${MRAY_SOURCE_DIRECTORY}/Core/Log.h
    ${MRAY_SOURCE_DIRECTORY}/Core/Vector.h
    ${MRAY_SOURCE_DIRECTORY}/Core/Error.h
    ${MRAY_SOURCE_DIRECTORY}/Core/Types.h
    ${MRAY_SOURCE_DIRECTORY}/Core/MRayDataType.h
    <tuple>
    <span>
    <optional>
    <variant>
    <vector>
    <map>
    <functional>
    <filesystem>
)

if(MSVC)
    set(SRC_HEADERS ${SRC_HEADERS}
        <Windows.h>)
endif()

add_library(GlobalPCH STATIC
            ${MRAY_SOURCE_DIRECTORY}/dummy_pch.cpp)
target_precompile_headers(GlobalPCH PRIVATE ${SRC_HEADERS})
target_include_directories(GlobalPCH PRIVATE ${MRAY_LIB_INCLUDE_DIRECTORY})
target_link_libraries(GlobalPCH PRIVATE
                      mray::meta_compile_opts
                      fmt::fmt)
set_target_properties(GlobalPCH PROPERTIES FOLDER
                      ${MRAY_PREDEFINED_TARGETS_FOLDER})
