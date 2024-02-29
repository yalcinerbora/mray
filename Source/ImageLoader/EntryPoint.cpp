
#include "EntryPoint.h"
#include "ImageLoader.h"

extern "C" MRAY_MESHLOADER_ENTRYPOINT
ImageLoaderI* ImageLoaderInstance()
{
    static std::unique_ptr<ImageLoader> instance = nullptr;
    if(instance == nullptr)
        instance = std::make_unique<ImageLoader>();

    return instance.get();
}