
#include "EntryPoint.h"
#include "ImageLoader.h"

extern "C" MRAY_IMAGELOADER_ENTRYPOINT
ImageLoaderI* ImageLoaderDetail::ConstructImageLoader(bool enableMT)
{
    return new ImageLoader(enableMT);
}

extern "C" MRAY_IMAGELOADER_ENTRYPOINT
void ImageLoaderDetail::DestroyImageLoader(ImageLoaderI* ptr)
{
    delete ptr;
}