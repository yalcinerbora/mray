include("GenTracerDLLTarget")

if(MRAY_ENABLE_HW_ACCELERATION)
    set(HW_ACCEL_FLAG ENABLE_HW_ACCEL)
endif()

gen_tracer_dll_target(NAME "CPU"
                      MACRO "MRAY_GPU_BACKEND_CPU"
                      ${HW_ACCEL_FLAG})

if(UNIX AND NOT APPLE)
    target_link_libraries(TracerDLL_CPU PRIVATE atomic)
endif()