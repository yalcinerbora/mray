#include "EntryPoint.h"
#include "SceneLoaderMRay.h"

// C Mangling, we will load these in runtime
extern "C" MRAY_SCENELOADER_MRAY_ENTRYPOINT
SceneLoaderI* SceneLoaderMRayDetail::ConstructSceneLoaderMRay()
{
    return new SceneLoaderMRay();
}

extern "C" MRAY_SCENELOADER_MRAY_ENTRYPOINT
void SceneLoaderMRayDetail::DestroySceneLoaderMRay(SceneLoaderI* ptr)
{
    if(ptr) delete ptr;
}