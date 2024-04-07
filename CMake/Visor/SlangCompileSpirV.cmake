# Similar approach related to this
# https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/nvcuda_compile_ptx.cmake

# Generate a custom build rule to translate a single *.slang file to *.spir-v file.
# slang_gen_spirv(
#   SOURCE file1.slang
#   EXTRA_OPTIONS <opions for nvcc> ...
# )
function(slang_gen_spriv)
    set(oneValueArgs SOURCE)
    set(multiValueArgs EXTRA_OPTIONS)

    cmake_parse_arguments(SLANG_GEN_SPIRV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Add default compile options
    set(SLANG_COMPILE_OPTIONS "")
    list(APPEND SLANG_COMPILE_OPTIONS ${SLANG_GEN_SPIRV_EXTRA_OPTIONS}

        -fvk-use-entrypoint-name
        -target spirv
        -fspv-reflect
        -emit-spirv-directly
        $<$<CONFIG:Debug>:-O1>
        $<$<CONFIG:Release>:-O3>
        -fp-mode $<$<CONFIG:Debug>:precise> $<$<CONFIG:Release>:fast>
        # Debug related preprocessor flags
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-DMRAY_DEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
    )

    add_custom_command(
        OUTPUT  "${OUTPUT}"
        COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
        MAIN_DEPENDENCY "${INPUT}"
        # Try c++ parsing mode maybe it works
        IMPLICIT_DEPENDS CXX "${INPUT}"
        COMMAND ${MRAY_SLANG_COMPILER} ${NVCC_COMPILE_OPTIONS}
                    "--gpu-architecture=${CC_FLAG}"
                    -o ${OUTPUT}
                    ${INPUT}
    )

endfunction()

# Generates *.spir-v files for the given source files.
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

function(NVCC_COMPILE_PTX)
    set(oneValueArgs GENERATED_TARGET MAIN_TARGET)
    set(multiValueArgs EXTRA_OPTIONS SOURCES)

    CMAKE_PARSE_ARGUMENTS(NVCC_COMPILE_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})



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
        # Debug related preprocessor flags
        $<$<CONFIG:Debug>:-DMRAY_DEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
        -DMRAY_CUDA
     )

    # Custom Target Name
    set(PTX_TARGET "${NVCC_COMPILE_PTX_MAIN_TARGET}_Optix")

    # Custom build rule to generate ptx files from cuda files
    FOREACH(INPUT ${NVCC_COMPILE_PTX_SOURCES})

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
        FOREACH(COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY_LIST})
            # Generate the *.ptx files to the appropirate bin directory.
            set(OUTPUT_STEM "${INPUT_STEM}.CC_${COMPUTE_CAPABILITY}")
            set(OUTPUT_FILE "${OUTPUT_STEM}.optixir")
            set(OUTPUT_DIR "${MRAY_CONFIG_BIN_DIRECTORY}/OptiXShaders")
            set(OUTPUT "${OUTPUT_DIR}/${OUTPUT_FILE}")

            set(DEP_FILE "${OUTPUT_STEM}.d")
            set(DEP_PATH "${OUTPUT_DIR}/${DEP_FILE}")

            list(APPEND PTX_FILES ${OUTPUT})

            set(CC_FLAG ${COMPUTE_CAPABILITY})
            if(${COMPUTE_CAPABILITY} MATCHES "^[0-9]+$")
                # If numbered add compute
                set(CC_FLAG compute_${COMPUTE_CAPABILITY})
            endif()

            # This prints the standalone NVCC command line for each CUDA file.
            #message(STATUS "${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS} ${INPUT} -o ${OUTPUT} -odir ${OUTPUT_DIR}")
            #message(STATUS ${NVCC_COMPILE_OPTIONS})
            add_custom_command(
                OUTPUT  "${OUTPUT}"
                COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
                MAIN_DEPENDENCY "${INPUT}"
                DEPFILE "${DEP_PATH}"
                COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
                         -MD
                         "--gpu-architecture=${CC_FLAG}"
                         -o ${OUTPUT}
                         ${INPUT}

                # TODO: Check that if this works
                # IMPLICIT_DEPENDS CXX "${INPUT}"
                # DEPFILE "${DEP_PATH}"
                # COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
                #         -M
                #         -o ${DEP_PATH}
                #         ${INPUT}
            )
        ENDFOREACH()
  ENDFOREACH()

  # Custom Target for PTX Files Main Target should depend on this target
  add_custom_target(${PTX_TARGET}
                    DEPENDS ${PTX_FILES}
                    # Add Source files for convenience
                    SOURCES ${NVCC_COMPILE_PTX_SOURCES}
                    )

  set(${NVCC_COMPILE_PTX_GENERATED_TARGET} ${PTX_TARGET} PARENT_SCOPE)
ENDFUNCTION()