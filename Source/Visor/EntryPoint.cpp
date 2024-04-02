#include "EntryPoint.h"
#include "Visor.h"

extern "C" MRAY_VISOR_ENTRYPOINT
VisorI* VisorDetail::ConstructVisor()
{
    return new VisorVulkan();
}

extern "C" MRAY_VISOR_ENTRYPOINT
void VisorDetail::DestroyVisor(VisorI* ptr)
{
    delete ptr;
}