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

function(nvcc_compile_ptx)
    set(oneValueArgs GENERATED_TARGET MAIN_TARGET)
    set(multiValueArgs EXTRA_OPTIONS SOURCES)

    cmake_parse_arguments(NVCC_COMPILE_PTX "${options}" "${oneValueArgs}"
                         "${multiValueArgs}" ${ARGN})

    # Add -- ptx and extra provided options
    # Main Compile Options as well form the system
    set(NVCC_COMPILE_OPTIONS "")
    # Linux wants this i dunno why
    if(UNIX)
        list(APPEND NVCC_COMPILE_OPTIONS --compiler-bindir=${CMAKE_CXX_COMPILER})
    endif()
    # Generic Options
    list(APPEND NVCC_COMPILE_OPTIONS ${NVCC_COMPILE_PTX_EXTRA_OPTIONS}
        # Include Directories
        "-I${OPTIX_INCLUDE_DIR}"
        "-I${MRAY_SOURCE_DIRECTORY}"
        "-I${MRAY_LIB_INCLUDE_DIRECTORY}"
        --optix-ir
        --machine=64
        "-std=c++${CMAKE_CUDA_STANDARD}"
        "--relocatable-device-code=true"
        "--keep-device-functions"
        # OptiX Documentation says that -G'ed kernels may fail
        # So -lineinfo is used on both configurations
        $<$<CONFIG:Debug>:-G>
        $<$<CONFIG:SanitizeR>:-lineinfo>
        $<$<CONFIG:Release>:-lineinfo>
        # Debug related preprocessor flags
        $<$<CONFIG:Debug>:-DMRAY_DEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:SanitizeR>:-DNDEBUG>
        -DMRAY_CUDA
     )

    # Custom Target Name
    set(PTX_TARGET "${NVCC_COMPILE_PTX_MAIN_TARGET}_Optix")

    # Custom build rule to generate ptx files from cuda files
    foreach(INPUT ${NVCC_COMPILE_PTX_SOURCES})

        get_filename_component(INPUT_STEM "${INPUT}" NAME_WE)

        if(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "all")
            # We need to manually set the list here
            set(COMPUTE_CAPABILITY_LIST
                50 52 53
                60 61 62
                70 72 75
                80 86 87 89
                90 90a)
        elseif(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "all-major")
            set(COMPUTE_CAPABILITY_LIST 50 60 70 80 90)
        else()
            # "native" or old school numbered style is selected do nothing
            set(COMPUTE_CAPABILITY_LIST ${CMAKE_CUDA_ARCHITECTURES})
        endif()

        # Generate New Ptx file for each CC Requested
        foreach(COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY_LIST})
            # Generate the *.ptx files to the appropirate bin directory.
            set(OUTPUT_STEM "${INPUT_STEM}.CC_${COMPUTE_CAPABILITY}")
            set(OUTPUT_FILE "${OUTPUT_STEM}.optixir")
            set(OUTPUT_DIR "${MRAY_CONFIG_BIN_DIRECTORY}/OptiXShaders")
            set(OUTPUT "${OUTPUT_DIR}/${OUTPUT_FILE}")



            list(APPEND PTX_FILES ${OUTPUT})

            set(CC_FLAG ${COMPUTE_CAPABILITY})
            if(${COMPUTE_CAPABILITY} MATCHES "^[0-9]+$")
                # If numbered add compute
                set(CC_FLAG compute_${COMPUTE_CAPABILITY})
            endif()

            # This prints the standalone NVCC command line for each CUDA file.
            add_custom_command(
                OUTPUT  "${OUTPUT}"
                COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
                MAIN_DEPENDENCY "${INPUT}"
                IMPLICIT_DEPENDS CXX "${INPUT}"
                COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
                         #-MD
                         "--gpu-architecture=${CC_FLAG}"
                         -o ${OUTPUT}
                         ${INPUT}
            )

            # TODO: Check that if this works now
            # This fails but dunno why? (relative path etc maybe?)
            # set(DEP_FILE "${OUTPUT_STEM}.d")
            # set(DEP_PATH "${OUTPUT_DIR}/${DEP_FILE}")
            # add_custom_command(
            #     OUTPUT  "${OUTPUT}"
            #     COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
            #     IMPLICIT_DEPENDS CXX "${INPUT}"
            #     DEPFILE "${DEP_PATH}"
            #     COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
            #             -MD
            #             -o ${DEP_PATH}
            #             ${INPUT}
            # )
        endforeach()
  endforeach()

  # Custom Target for PTX Files Main Target should depend on this target
  add_custom_target(${PTX_TARGET}
                    DEPENDS ${PTX_FILES}
                    # Add Source files for convenience
                    SOURCES ${NVCC_COMPILE_PTX_SOURCES}
                    )

  set(${NVCC_COMPILE_PTX_GENERATED_TARGET} ${PTX_TARGET} PARENT_SCOPE)
endfunction()