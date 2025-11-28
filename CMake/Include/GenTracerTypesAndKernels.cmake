

function(tracer_kernel_gen_file_names)
    set(options)
    set(oneValueArgs NUM_FILES)
    set(multiValueArgs)

    cmake_parse_arguments(TRACER_KERNEL_GEN
                          "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    # THESE MUST MATCH WITH THE
    # CODE IN THE TracerKernelGen EXECUTABLE
    # TODO: Create configure file for these
    list(APPEND KERNEL_GEN_FILES
         "_GEN_RequestedTypes.h"
         "_GEN_RequestedRenderers.h")
    foreach(i RANGE 1 ${TRACER_KERNEL_GEN_NUM_FILES})
        list(APPEND KERNEL_GEN_INSTANTIATIONS "_GEN_RendererKernels${i}.cu")
    endforeach()

    list(APPEND KERNEL_GEN_INSTANTIATIONS "_GEN_CommonKernels.cu")

    set(TRACER_KERNEL_GEN_HEADERS ${KERNEL_GEN_FILES} PARENT_SCOPE)
    set(TRACER_KERNEL_GEN_INSTANTIATIONS
        ${KERNEL_GEN_INSTANTIATIONS} PARENT_SCOPE)
endfunction()