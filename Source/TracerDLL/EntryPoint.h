#pragma once

#include "Core/TracerI.h"
#include "Core/System.h"

#ifdef MRAY_TRACER_DEVICE_SHARED_EXPORT
    #define MRAY_TRACER_DEVICE_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_TRACER_DEVICE_ENTRYPOINT MRAY_DLL_IMPORT
#endif

// C Interface (Used when dynamically loading the DLL)
extern "C" MRAY_TRACER_DEVICE_ENTRYPOINT
TracerI* ConstructTracer(const TracerParameters&);

extern "C" MRAY_TRACER_DEVICE_ENTRYPOINT
void DestroyTracer(TracerI*);