#pragma once

#include <memory>
#include "ImageLoaderI.h"
#include "Core/System.h"
#include "Core/SharedLibrary.h"

#ifdef MRAY_IMAGELOADER_SHARED_EXPORT
    #define MRAY_IMAGEOADER_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_IMAGEOADER_ENTRYPOINT MRAY_DLL_IMPORT
#endif

// C Interface (Used when dynamically loading the DLL)
namespace ImageLoaderDetail
{
    extern "C" MRAY_IMAGEOADER_ENTRYPOINT
    ImageLoaderI* ConstructImageLoader();

    extern "C" MRAY_IMAGEOADER_ENTRYPOINT
    void DestroyImageLoader(ImageLoaderI*);

}

// C++ Interface
inline
std::unique_ptr<ImageLoaderI, decltype(&ImageLoaderDetail::DestroyImageLoader)>
CreateImageLoader()
{
    using Ptr = std::unique_ptr<ImageLoaderI, decltype(&ImageLoaderDetail::DestroyImageLoader)>;
    return Ptr(ImageLoaderDetail::ConstructImageLoader(),
               &ImageLoaderDetail::DestroyImageLoader);
}