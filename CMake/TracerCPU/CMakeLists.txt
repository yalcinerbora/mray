include("GenTracerDLLTarget")

gen_tracer_dll_target(NAME "CPU"
                      MACRO "MRAY_GPU_BACKEND_CPU")

if(UNIX AND NOT APPLE)
    target_link_libraries(TracerDLL_CPU PRIVATE atomic)
endif()