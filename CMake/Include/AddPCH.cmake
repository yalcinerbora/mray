
function(add_precompiled_headers)

    if(NOT MRAY_ENABLE_PCH)
        return()
    endif()

    set(options)
    set(oneValueArgs NAME TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(GENERATE_PRECOMPILED_HEADERS "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    # TODO: Global PCH does not work on MSVC
    # debug symbol related issuue
    # currently we utilize per target PCH
    # target_precompile_headers(${GENERATE_PRECOMPILED_HEADERS_TARGET}
    #                          REUSE_FROM GlobalPCH)

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

    target_precompile_headers(${GENERATE_PRECOMPILED_HEADERS_TARGET}
                            PRIVATE ${SRC_HEADERS})

endfunction()