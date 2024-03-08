
#include "EntryPoint.h"
#include "ImageLoader.h"

extern "C" MRAY_IMAGEOADER_ENTRYPOINT
ImageLoaderI* ImageLoaderDetail::ConstructImageLoader()
{
    return new ImageLoader();
}

extern "C" MRAY_IMAGEOADER_ENTRYPOINT
void ImageLoaderDetail::DestroyImageLoader(ImageLoaderI* ptr)
{
    delete ptr;
}