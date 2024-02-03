
#include "Core/Types.h"

#include <iostream>

//#include "SceneLoaderMRay/EntryPoint.h"

#include "Core/SharedLibrary.h"
#include "Core/SceneLoaderI.h"

#include <BS/BS_thread_pool.hpp>

int main()
{
    BS::thread_pool pool;

    SharedLibrary lib("SceneLoaderMRay");

    SharedLibPtr<SceneLoaderI> loader(nullptr, nullptr);
    MRayError e = lib.GenerateObjectWithArgs(loader,
                                             SharedLibArgs{"ConstructSceneLoaderMRay",
                                                           "DestroySceneLoaderMRay"},
                                             pool);

    if(e) MRAY_ERROR_LOG("{}", e.GetError());

    //loader->LoadScene("Scenes/Kitchen/Kitchen.json");

    // Load Args
    // Load visor if requested

    // Load tracer from DLL
    // Load scene loader

    //ask scene loader to load a scene
}