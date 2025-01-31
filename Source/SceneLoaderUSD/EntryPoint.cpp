#include "EntryPoint.h"
#include "SceneLoaderUSD.h"

// C Mangling, we will load these in runtime
extern "C" MRAY_SCENELOADER_USD_ENTRYPOINT
SceneLoaderI* SceneLoaderMRayDetail::ConstructSceneLoaderUSD(BS::thread_pool& p)
{
    return new SceneLoaderUSD(p);
}

extern "C" MRAY_SCENELOADER_USD_ENTRYPOINT
void SceneLoaderMRayDetail::DestroySceneLoaderUSD(SceneLoaderI* ptr)
{
    if(ptr) delete ptr;
}