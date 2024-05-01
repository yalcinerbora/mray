#include "EntryPoint.h"
#include "Tracer.h"

extern "C" MRAY_TRACER_DEVICE_ENTRYPOINT
TracerI* ConstructTracer(BS::thread_pool& tp)
{
    return new TracerBase(tp);
}

extern "C" MRAY_TRACER_DEVICE_ENTRYPOINT
void DestroyTracer(TracerI* ptr)
{
    delete ptr;
}