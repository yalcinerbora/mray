include("GenTracerDLLTarget")

if(MRAY_ENABLE_HW_ACCELERATION)
    set(HW_ACCEL_FLAG ENABLE_HW_ACCEL)
endif()

gen_tracer_dll_target(NAME ${MRAY_GPU_PLATFORM_NAME}
                      MACRO ${MRAY_DEVICE_BACKEND}
                      ${HW_ACCEL_FLAG})

