# Similar approach related to this
# https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/nvcuda_compile_ptx.cmake

set(MRAY_SPIRV_EXTENSION ".spv")
set(MRAY_SPIRV_ASM_EXTENSION ".spv-asm")

# Generate a custom build rule to create *.slang-module from given files.
# Slang module files implicitly stored on the "${TARGET_LOCAL_SHADER_MODULE_OUT_DIRECTORY}".
#
#   OUTPUT <output file name>             : extensionless output module name
#   SOURCES <file1> <file2> ...           : source files to compile (with .slang extension)
#   INCLUDES <include1> <include2> ...    : include directories (should not have -I prefix)
#   EXTRA_OPTIONS <option1> <option2> ... : additional options for the compiler
function(slang_gen_module)
    set(oneValueArgs OUTPUT)
    set(multiValueArgs EXTRA_OPTIONS SOURCES INCLUDES)
    cmake_parse_arguments(SLANG_GEN_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(TRANSFORM SLANG_GEN_MODULE_INCLUDES PREPEND "-I")

    # Add default compile options
    set(SLANG_COMPILE_OPTIONS "")
    list(APPEND SLANG_COMPILE_OPTIONS ${SLANG_GEN_MODULE_EXTRA_OPTIONS}
        # Config related preprocessor flags
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O1>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-DMRAY_DEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
        ${SLANG_GEN_MODULE_INCLUDES}
    )
    set(MODULE_NAME "${SLANG_GEN_MODULE_OUTPUT}.slang-module")
    set(MODULE_OUTPUT_PATH "${MRAY_SHADER_MODULE_OUT_DIRECTORY}/${MODULE_NAME}")

    add_custom_command(
        OUTPUT  ${MODULE_OUTPUT_PATH}
        COMMENT "[SHADER] Building slang-module \"${MODULE_NAME}\""
        DEPENDS ${SLANG_GEN_MODULE_SOURCES}
        COMMAND ${MRAY_SLANG_COMPILER} ${SLANG_COMPILE_OPTIONS}
                -o ${MODULE_OUTPUT_PATH}
                ${SLANG_GEN_MODULE_SOURCES}
    )

endfunction()


# Generate a custom build rule to create *.spv and optionally *.spv-asm from given module group.
# Support type generation semantics via slang linking.
# Spir-V files implicitly stored on the "${MRAY_SHADER_OUT_DIRECTORY}".
#
#   OUTPUT_PREFIX <output>                : extensionless output file prefix, it will be concatanated
#                                         : with preprocessor macro
#   MODULES <file1> <file2> ...           : module files to compile (without extension, and
#                                         : without a path) Path is implicitly "${MRAY_SHADER_MODULE_OUT_DIRECTORY}"
#   INCLUDES <include1> <include2> ...    : include directories (should not have -I prefix)
#   EXTRA_OPTIONS <option1> <option2> ... : additional options for the compiler
#
#   TYPEGEN_SHADER_FILE                   : a slang file that exports the types. May be guarded by a macro
#   TYPEGEN_MACRO                         : a macro that guards the type generation will be fed to the slang.
#   GEN_ASSEMBLY                          : Generate both spirv binary and assembly
function(slang_gen_spriv)
    set(oneValueArgs
        OUTPUT_PREFIX
        TYPEGEN_SHADER_FILE
        TYPEGEN_MACRO)
    set(multiValueArgs
        EXTRA_OPTIONS
        MODULES
        INCLUDES
        DEPENDS)
    set(options GEN_ASSEMBLY)
    cmake_parse_arguments(SLANG_GEN_SPIRV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(TRANSFORM SLANG_GEN_SPIRV_INCLUDES PREPEND "-I")
    list(TRANSFORM SLANG_GEN_SPIRV_MODULES APPEND ".slang-module")
    list(TRANSFORM SLANG_GEN_SPIRV_MODULES PREPEND "${MRAY_SHADER_MODULE_OUT_DIRECTORY}/")

    set(OUTPUT_NAME "${SLANG_GEN_SPIRV_OUTPUT_PREFIX}-${SLANG_GEN_SPIRV_TYPEGEN_MACRO}")
    slang_gen_module(OUTPUT "${OUTPUT_NAME}"
                     SOURCES ${SLANG_GEN_SPIRV_TYPEGEN_SHADER_FILE}
                     EXTRA_OPTIONS "-D${SLANG_GEN_SPIRV_TYPEGEN_MACRO}")

    # Add default compile options
    set(SLANG_COMPILE_OPTIONS "")
    list(APPEND SLANG_COMPILE_OPTIONS ${SLANG_GEN_SPIRV_EXTRA_OPTIONS}
        -fvk-use-entrypoint-name
        -capability sm_6_0
        -emit-spirv-directly
        -fp-mode $<$<CONFIG:Debug>:precise> $<$<CONFIG:Release>:fast>
        # Config related preprocessor flags
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O1>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-DMRAY_DEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
        ${SLANG_GEN_SPIRV_INCLUDES}
    )

    #set(OUTPUT_FILE "${OUTPUT_NAME}.spv-asm")
    set(TYPE_MODULE "${MRAY_SHADER_MODULE_OUT_DIRECTORY}/${OUTPUT_NAME}.slang-module")
    list(APPEND SPIRV_DEPS ${SLANG_GEN_SPIRV_DEPENDS} ${TYPE_MODULE} ${SLANG_GEN_SPIRV_MODULES})
    set(SPIRV_OUTPUT_PATH "${MRAY_SHADER_OUT_DIRECTORY}/${OUTPUT_NAME}")

    message(STATUS ${SLANG_GEN_SPIRV_GEN_ASSEMBLY})

    set(DEPENDENCY_LIST)
    if(SLANG_GEN_SPIRV_GEN_ASSEMBLY)
        set(OUT_SPV_ASM ${SPIRV_OUTPUT_PATH}${MRAY_SPIRV_ASM_EXTENSION})
        add_custom_command(
            OUTPUT  ${OUT_SPV_ASM}
            COMMENT "[SHADER] Building spir-V assembly \"${OUTPUT_NAME}${MRAY_SPIRV_ASM_EXTENSION}\""
            DEPENDS ${SPIRV_DEPS}
            COMMAND ${MRAY_SLANG_COMPILER} ${SLANG_COMPILE_OPTIONS}
                    -o ${OUT_SPV_ASM}
                    ${SLANG_GEN_SPIRV_MODULES}
                    ${TYPE_MODULE})

        message(STATUS ${OUT_SPV_ASM})
        set(DEPENDENCY_LIST ${DEPENDENCY_LIST} ${OUT_SPV_ASM})

    endif()

    set(OUT_SPV ${SPIRV_OUTPUT_PATH}${MRAY_SPIRV_EXTENSION})
    add_custom_command(
        OUTPUT  ${OUT_SPV}
        COMMENT "[SHADER] Building spir-V \"${OUTPUT_NAME}${MRAY_SPIRV_EXTENSION}\""
        DEPENDS ${SPIRV_DEPS}
        COMMAND ${MRAY_SLANG_COMPILER} ${SLANG_COMPILE_OPTIONS}
                -o ${OUT_SPV}
                ${SLANG_GEN_SPIRV_MODULES}
                ${TYPE_MODULE})

    set(DEPENDENCY_LIST ${DEPENDENCY_LIST} ${OUT_SPV})
    set(SLANG_GEN_SPIRV_OUTPUT_LIST ${SLANG_GEN_SPIRV_OUTPUT_LIST}
                ${DEPENDENCY_LIST} PARENT_SCOPE)

endfunction()
