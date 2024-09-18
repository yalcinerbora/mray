# Had as imilar approach related to this
# https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/nvcuda_compile_ptx.cmake
#
# Generate a custom build rule to translate *.cu files to *.ptx files.
# nvcc_compile_ptx(
#   MAIN_TARGET TargetName
#   SOURCES file1.cu file2.cu ...
#   GENERATED_TARGET <generated target, (variable stores TargetName_Optix)>
#   EXTRA_OPTIONS <opions for nvcc> ...
# )
#
# It has issues with automatic dependency generation, luckily
# we can use CMake's built-in functionality to generate the IR files
# which is introduced in CMake 3.27.
#
# New function, we create a target which emits optix-ir
# however when the system is compiled with multiple CC's this approach do not work.
# So we create 1 target for each CC and create another single target to
# copy these near executable's location.
#
# Last single target is required because we can not use add_custom_command(TARGET ...)
# with object libraries.
#
# Generate a custom build rule to translate *.cu files to *.ptx files.
# nvcc_compile_optix_ir(
#   TARGET_PREFIX name
#   SOURCES file1.cu file2.cu ...
# )
# This function implictly depends on the following variables:
# - OPTIX_INCLUDE_DIR           (to include the optix headers)
# - MRAY_CONFIG_BIN_DIRECTORY   (to copy the generated ir files files)
# - MRAY_GPU_PLATFORM_NAME      (to organize the generated files)
# - CMAKE_CUDA_ARCHITECTURES    (to determine the compute capabilities)
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
        # target_compile_definitions(${TARGET_NAME}
        #                            PUBLIC
        #                            MRAY_OPTIX)

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