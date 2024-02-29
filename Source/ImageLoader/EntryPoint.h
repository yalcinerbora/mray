#pragma once

#include "ImageLoaderI.h"
#include "Core/System.h"
#include "Core/SharedLibrary.h"

#ifdef MRAY_IMAGELOADER_SHARED_EXPORT
    #define MRAY_MESHLOADER_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_MESHLOADER_ENTRYPOINT MRAY_DLL_IMPORT
#endif

extern "C" MRAY_MESHLOADER_ENTRYPOINT
ImageLoaderI* ImageLoaderInstance();