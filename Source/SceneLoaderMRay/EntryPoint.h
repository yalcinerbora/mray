#pragma once

#include "Core/System.h"
#include "Core/SharedLibrary.h"
#include "Core/SceneLoaderI.h"

#ifdef MRAY_SCENELOADER_MRAY_SHARED_EXPORT
#define MRAY_SCENELOADER_MRAY_ENTRYPOINT MRAY_DLL_EXPORT
#else
#define MRAY_SCENELOADER_MRAY_ENTRYPOINT MRAY_DLL_IMPORT
#endif

// C Mangling, we will load these in runtime
namespace SceneLoaderMRayDetail
{

extern "C" MRAY_SCENELOADER_MRAY_ENTRYPOINT
SceneLoaderI* ConstructSceneLoaderMRay();

extern "C" MRAY_SCENELOADER_MRAY_ENTRYPOINT
void DestroySceneLoaderMRay(SceneLoaderI* ptr);

}