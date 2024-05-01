#include "EntryPoint.h"
#include "Tracer.h"

extern "C" MRAY_TRACER_DEVICE_ENTRYPOINT
TracerI* ConstructTracer()
{
    return new TracerBase();
}

extern "C" MRAY_TRACER_DEVICE_ENTRYPOINT
void DestroyTracer(TracerI* ptr)
{
    delete ptr;
}