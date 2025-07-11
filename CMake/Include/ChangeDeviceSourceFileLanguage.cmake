function(change_device_source_file_language)

    # Parse Args
    set(options)
    set(oneValueArgs MACRO)
    set(multiValueArgs SOURCE_FILES)

    cmake_parse_arguments(CHANGE_SOURCE_LANG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(CHANGE_SOURCE_LANG_MACRO STREQUAL "MRAY_GPU_BACKEND_CUDA")
        set(CHANGE_SOURCE_LANG_NAME CUDA)
    elseif(CHANGE_SOURCE_LANG_MACRO STREQUAL "MRAY_GPU_BACKEND_HIP")
        set(CHANGE_SOURCE_LANG_NAME HIP)
    elseif(CHANGE_SOURCE_LANG_MACRO STREQUAL "MRAY_GPU_BACKEND_CPU")
        set(CHANGE_SOURCE_LANG_NAME CXX)
    else()
        message(FATAL_ERROR "Unknown MRAY_GPU_BACKEND_* macro!")
    endif()

    set_source_files_properties(${CHANGE_SOURCE_LANG_SOURCE_FILES}
                                PROPERTIES LANGUAGE
                                ${CHANGE_SOURCE_LANG_NAME})

endfunction()