#pragma once

#include <memory>

#include "Core/System.h"
#include "VisorI.h"

#ifdef MRAY_VISOR_SHARED_EXPORT
    #define MRAY_VISOR_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_VISOR_ENTRYPOINT MRAY_DLL_IMPORT
#endif

namespace VisorDetail
{
    extern "C" MRAY_VISOR_ENTRYPOINT
    VisorI* ConstructVisor();

    extern "C" MRAY_VISOR_ENTRYPOINT
    void    DestroyVisor(VisorI*);
}

// C++ Interface
inline
std::unique_ptr<VisorI, decltype(&VisorDetail::DestroyVisor)>
CreateVisor()
{
    using namespace VisorDetail;
    using Ptr = std::unique_ptr<VisorI, decltype(&DestroyVisor)>;
    return Ptr(ConstructVisor(), &DestroyVisor);
}