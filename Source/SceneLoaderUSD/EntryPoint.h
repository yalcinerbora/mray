#pragma once

#include "Core/System.h"

#ifdef MRAY_SCENELOADER_USD_SHARED_EXPORT
#define MRAY_SCENELOADER_USD_ENTRYPOINT MRAY_DLL_EXPORT
#else
#define MRAY_SCENELOADER_USD_ENTRYPOINT MRAY_DLL_IMPORT
#endif

class SceneLoaderI;
class ThreadPool;

// C Mangling, we will load these in runtime
namespace SceneLoaderMRayDetail
{

extern "C" MRAY_SCENELOADER_USD_ENTRYPOINT
SceneLoaderI* ConstructSceneLoaderUSD(ThreadPool&);

extern "C" MRAY_SCENELOADER_USD_ENTRYPOINT
void DestroySceneLoaderUSD(SceneLoaderI* ptr);

}