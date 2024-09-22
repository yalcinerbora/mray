include("NVCCCompilePTX")

# Calling function from this scope
# source_group etc. are lost, it is good so we can put these on root
# scope
source_group("" FILES ${SRC_ACCELLERATORS_PTX})

nvcc_compile_optix(TARGET_PREFIX ${TARGET_FULL_NAME}
                   SOURCES ${SRC_ACCELLERATORS_PTX}
                   AS_PTX ${MRAY_COMPILE_OPTIX_AS_PTX})

add_dependencies(${TARGET_FULL_NAME} ${NVCC_COMPILE_OPTIX_GENERATED_TARGETS})