# Similar approach related to this
# https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/nvcuda_compile_ptx.cmake

# Generate a custom build rule to translate *.cu files to *.ptx files.
# nvcc_compile_ptx(
#   MAIN_TARGET TargetName
#   SOURCES file1.cu file2.cu ...
#   GENERATED_TARGET <generated target, (variable stores TargetName_Optix)>
#   EXTRA_OPTIONS <opions for nvcc> ...
# )

# Generates *.ptx files for the given source files.
# Unlike the copied code;
#    It also generates a custom target for files since I did not want to see
#    PTX output on the Visual Studio.
#
#    It returns the generated target for depedency setting.
#
#    It tries to set the dependencies automatically (dunno if this works tho)
#
#    Additionally it generates different PTX for each Compute Capability defined
#    in CMAKE_CUDA_ARCHITECTURES variable
#
#    Finally It also outputs as <filename>_CC[50,61..].o.ptx
#    because it is a good pun =) (Normally it was goint to be optx but maybe some
#    files)
#    Unfortunately this is changed it is now .optixir =(, and only optixir is used
#    only (7.5 or above)
#    TODO: change this to make it compatible with older versions of the optix
# function(nvcc_compile_ptx)
#     set(oneValueArgs MAIN_TARGET)
#     set(multiValueArgs EXTRA_OPTIONS SOURCES DEPS MACROS)

#     cmake_parse_arguments(NVCC_COMPILE_PTX "${options}" "${oneValueArgs}"
#                          "${multiValueArgs}" ${ARGN})

#     list(TRANSFORM NVCC_COMPILE_PTX_MACROS PREPEND "-D")

#     # Add -- ptx and extra provided options
#     # Main Compile Options as well form the system
#     set(NVCC_COMPILE_OPTIONS "")
#     # Linux wants this i dunno why
#     if(UNIX)
#         list(APPEND NVCC_COMPILE_OPTIONS --compiler-bindir=${CMAKE_CXX_COMPILER})
#     endif()
#     # Generic Options
#     list(APPEND NVCC_COMPILE_OPTIONS ${NVCC_COMPILE_PTX_EXTRA_OPTIONS}
#         # Include Directories
#         -I${OPTIX_INCLUDE_DIR}
#         -I${MRAY_SOURCE_DIRECTORY}
#         -I${MRAY_LIB_INCLUDE_DIRECTORY}
#         --optix-ir
#         --machine=64
#         -std=c++${CMAKE_CUDA_STANDARD}
#         --relocatable-device-code=true
#         --keep-device-functions
#         --expt-relaxed-constexpr
#         # OptiX Documentation says that -G'ed kernels may fail
#         # So -lineinfo is used on both configurations
#         $<$<CONFIG:Debug>:-lineinfo>
#         $<$<CONFIG:SanitizeR>:-lineinfo>
#         $<$<CONFIG:Release>:-lineinfo>
#         # Debug related preprocessor flags
#         $<$<CONFIG:Debug>:-DMRAY_DEBUG>
#         $<$<CONFIG:Release>:-DNDEBUG>
#         $<$<CONFIG:SanitizeR>:-DNDEBUG>
#         ${NVCC_COMPILE_PTX_MACROS}
#      )

#     # Custom Target Name
#     set(PTX_TARGET "${NVCC_COMPILE_PTX_MAIN_TARGET}_OptiX-IR")

#     # Custom build rule to generate ptx files from cuda files
#     foreach(INPUT ${NVCC_COMPILE_PTX_SOURCES})

#         # Skip source files that are not cuda files (probably headers)
#         cmake_path(GET INPUT EXTENSION  INPUT_EXT)
#         if(NOT ${INPUT_EXT} STREQUAL ".cu")
#             continue()
#         endif()

#         cmake_path(GET INPUT STEM INPUT_STEM)

#         if(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "all")
#             # TODO: I think we can get these programatically
#             # but it is a hassle, so we will just set them manually
#             set(COMPUTE_CAPABILITY_LIST
#                 50 52 53
#                 60 61 62
#                 70 72 75
#                 80 86 87 89
#                 90 90a)
#         elseif(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "all-major")
#             set(COMPUTE_CAPABILITY_LIST 50 60 70 80 90)
#         else()
#             # "native" is selected, do nothing
#             set(COMPUTE_CAPABILITY_LIST ${CMAKE_CUDA_ARCHITECTURES})
#         endif()

#         # CC Common depfile
#         set(DEP_FILE "${INPUT_STEM}.cu.d")
#         add_custom_command(
#                 OUTPUT  "${DEP_FILE}"
#                 COMMENT "Builidng Depfile of ${INPUT}"
#                 MAIN_DEPENDENCY ${INPUT}
#                 COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
#                         -M
#                         -o ${DEP_FILE}
#                         ${INPUT}
#             )

#         list(APPEND DEP_FILES ${DEP_FILE})

#         # Generate New Ptx file for each CC Requested
#         foreach(COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY_LIST})
#             # Generate the *.ptx files to the appropirate bin directory.
#             set(OUTPUT_STEM "${INPUT_STEM}.CC_${COMPUTE_CAPABILITY}")
#             set(OUTPUT_FILE "${OUTPUT_STEM}.optixir")
#             set(OUTPUT_DIR "${MRAY_CONFIG_BIN_DIRECTORY}/OptiXShaders")
#             set(OUTPUT "${OUTPUT_DIR}/${OUTPUT_FILE}")

#             list(APPEND PTX_FILES ${OUTPUT})

#             set(CC_FLAG ${COMPUTE_CAPABILITY})
#             if(${COMPUTE_CAPABILITY} MATCHES "^[0-9]+$")
#                 # If numbered add compute
#                 set(CC_FLAG compute_${COMPUTE_CAPABILITY})
#             endif()

#             # This prints the standalone NVCC command line for each CUDA file.
#             # add_custom_command(
#             #     OUTPUT  "${OUTPUT}"
#             #     COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
#             #     MAIN_DEPENDENCY "${INPUT}"
#             #     DEPENDS ${NVCC_COMPILE_PTX_DEPS}
#             #     COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
#             #              "--gpu-architecture=${CC_FLAG}"
#             #              -o ${OUTPUT}
#             #              ${INPUT}
#             # )

#             # TODO: Check that if this works now
#             # This fails but dunno why? (relative path etc maybe?)
#             # set(DEP_FILE "${OUTPUT}.d")
#             # add_custom_command(
#             #     OUTPUT  "${DEP_FILE}"
#             #     COMMENT "Builidng Depfile of (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
#             #     MAIN_DEPENDENCY ${INPUT}
#             #     COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
#             #             -M
#             #             -o ${DEP_FILE}
#             #             ${INPUT}
#             # )
#             # add_custom_command(
#             #     OUTPUT  "${OUTPUT}"
#             #     COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
#             #     MAIN_DEPENDENCY ${INPUT}
#             #     DEPFILE ${DEP_FILE}
#             #     DEPENDS ${DEP_FILE}
#             #     COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
#             #              "--gpu-architecture=${CC_FLAG}"
#             #              -o ${OUTPUT}
#             #              ${INPUT}
#             # )
#         endforeach()
#   endforeach()


#     message(STATUS "Dep Files: ${DEP_FILES}")
#   # Custom Target for PTX Files Main Target should depend on this target
#   add_custom_target(${PTX_TARGET} DEPENDS ${PTX_FILES} ${DEP_FILES}
#                     SOURCES ${CMAKE_CURRENT_FUNCTION_LIST_FILE})

#   set(NVCC_COMPILE_PTX_GENERATED_TARGET ${PTX_TARGET} PARENT_SCOPE)
# endfunction()

# New function, we utilize CMake's functionality to generate the IR files
# This is better on windows since when the file is changed
# the dependencies auto generated (IMPLICIT_DEPENDS only works on makefile)
# I could not manage to make the DEP_FILE to work
function(nvcc_compile_optix_ir)

    set(oneValueArgs TARGET_PREFIX)
    set(multiValueArgs SOURCES)

    cmake_parse_arguments(NVCC_COMPILE_OPTIX_IR "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
    set(OPTIX_TARGET_NAME ${NVCC_COMPILE_OPTIX_IR_TARGET_PREFIX}_OptiXIR)

    # Here we need to do create a target for ALL CCs
    # This is the drawback of this approach
     if(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "all")
        # TODO: I think we can get these programatically
        # but it is a hassle, so we will just set them manually
        set(COMPUTE_CAPABILITY_LIST
            50 52 53
            60 61 62
            70 72 75
            80 86 87 89
            90 90a)
    elseif(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "all-major")
        set(COMPUTE_CAPABILITY_LIST 50 60 70 80 90)
    elseif(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "native")
        # Query the native compute capability via nvidia-smi
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE NATIVE_CC)
            string(REPLACE "." "" NATIVE_CC ${NATIVE_CC})
            string(STRIP ${NATIVE_CC} NATIVE_CC)
            set(COMPUTE_CAPABILITY_LIST ${NATIVE_CC})
    else()
        # a single number is selected, directly feed
        set(COMPUTE_CAPABILITY_LIST ${CMAKE_CUDA_ARCHITECTURES})
    endif()

    # Create the targets
    foreach(COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY_LIST})

        set(TARGET_NAME ${OPTIX_TARGET_NAME}_${COMPUTE_CAPABILITY})
        add_library(${TARGET_NAME} OBJECT
                    ${NVCC_COMPILE_OPTIX_IR_SOURCES})
        # Classic target stuff
        set_target_properties(${TARGET_NAME} PROPERTIES
                                CUDA_OPTIX_COMPILATION ON
                                CUDA_SEPARABLE_COMPILATION ON
                                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                CUDA_ARCHITECTURES "${COMPUTE_CAPABILITY}")
        #
        target_link_libraries(${TARGET_NAME}
                              PRIVATE
                              ${DEVICE_TARGET_FULL_NAME}
                              mray::meta_compile_opts
                              mray::cuda_extra_compile_opts)
        #
        target_include_directories(${TARGET_NAME}
                                   PRIVATE
                                   ${OPTIX_INCLUDE_DIR})
        target_compile_definitions(${TARGET_NAME}
                                   PUBLIC
                                   MRAY_OPTIX)

        set_target_properties(${TARGET_NAME} PROPERTIES
                              FOLDER ${MRAY_GPU_PLATFORM_NAME}/OptiX)

        # I could not figure to use generator expressions here
        # so we write to a different folder for each CC
        set(OUTPUT "${MRAY_CONFIG_BIN_DIRECTORY}/OptiXShaders/CC_${COMPUTE_CAPABILITY}")
        list(APPEND OUTPUT_FOLDERS ${OUTPUT})
        list(APPEND OUTPUT_FILES "$<TARGET_OBJECTS:${TARGET_NAME}>")
        list(APPEND OUTPUT_TARGETS ${TARGET_NAME})
    endforeach()

    # Since Object Libraries do not have add_custom_command(TARGET ...)
    # commands we create a custom target and do that operation
    set(OPTIX_SHADER_OUTPUT_DIR "${MRAY_CONFIG_BIN_DIRECTORY}/OptiXShaders")
    add_custom_target(OptiX_Copy
        DEPENDS ${OUTPUT_TARGETS}
        COMMENT "Copying OptiX IR Files to ${OPTIX_SHADER_OUTPUT_DIR}")
    foreach(num IN ZIP_LISTS OUTPUT_FILES OUTPUT_FOLDERS)
        add_custom_command(TARGET OptiX_Copy
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${num_1}
            COMMAND ${CMAKE_COMMAND} -E copy ${num_0} ${num_1}
            COMMAND_EXPAND_LISTS)
    endforeach()

    set_target_properties(OptiX_Copy PROPERTIES
                          FOLDER ${MRAY_GPU_PLATFORM_NAME}/OptiX)

    add_dependencies(OptiX_Copy ${OUTPUT_TARGETS})

    set(NVCC_COMPILE_OPTIX_IR_GENERATED_TARGETS OptiX_Copy PARENT_SCOPE)
endfunction()