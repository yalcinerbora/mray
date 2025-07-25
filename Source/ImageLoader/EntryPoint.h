#pragma once

#include <memory>
#include "ImageLoaderI.h"
#include "Core/System.h"

#ifdef MRAY_IMAGELOADER_SHARED_EXPORT
    #define MRAY_IMAGELOADER_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_IMAGELOADER_ENTRYPOINT MRAY_DLL_IMPORT
#endif

// C Interface (Used when dynamically loading the DLL)
namespace ImageLoaderDetail
{
    extern "C" MRAY_IMAGELOADER_ENTRYPOINT
    ImageLoaderI* ConstructImageLoader(bool enableMT = false);

    extern "C" MRAY_IMAGELOADER_ENTRYPOINT
    void DestroyImageLoader(ImageLoaderI*);

}

// C++ Interface
using ImageLoaderIPtr = std::unique_ptr<ImageLoaderI, decltype(&ImageLoaderDetail::DestroyImageLoader)>;

inline ImageLoaderIPtr CreateImageLoader(bool enableMT = false)
{
    using Ptr = std::unique_ptr<ImageLoaderI, decltype(&ImageLoaderDetail::DestroyImageLoader)>;
    return Ptr(ImageLoaderDetail::ConstructImageLoader(enableMT),
               &ImageLoaderDetail::DestroyImageLoader);
}