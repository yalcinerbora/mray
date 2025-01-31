#pragma once

#include "Core/System.h"
#include "Core/SharedLibrary.h"
#include "Core/SceneLoaderI.h"

#ifdef MRAY_SCENELOADER_USD_SHARED_EXPORT
#define MRAY_SCENELOADER_USD_ENTRYPOINT MRAY_DLL_EXPORT
#else
#define MRAY_SCENELOADER_USD_ENTRYPOINT MRAY_DLL_IMPORT
#endif

namespace BS
{
    class thread_pool;
}

// C Mangling, we will load these in runtime
namespace SceneLoaderMRayDetail
{

extern "C" MRAY_SCENELOADER_USD_ENTRYPOINT
SceneLoaderI* ConstructSceneLoaderUSD(BS::thread_pool&);

extern "C" MRAY_SCENELOADER_USD_ENTRYPOINT
void DestroySceneLoaderUSD(SceneLoaderI* ptr);

}