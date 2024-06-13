#include "TextureCUDA.h"

namespace mray::cuda
{

TextureBackingMemoryCUDA::TextureBackingMemoryCUDA(const GPUDeviceCUDA& gpu)
    : gpu(&gpu)
    , size(0)
    , allocSize(0)
    , memHandle(0)
{}

TextureBackingMemoryCUDA::TextureBackingMemoryCUDA(const GPUDeviceCUDA& device, size_t sizeInBytes)
    : TextureBackingMemoryCUDA(device)
{
    assert(sizeInBytes != 0);
    size = sizeInBytes;

    CUmemAllocationProp props = {};
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = gpu->DeviceId();
    props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;

    size_t granularity;
    CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(&granularity, &props,
                                                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    allocSize = MathFunctions::NextMultiple(size, granularity);
    CUDA_DRIVER_CHECK(cuMemCreate(&memHandle, allocSize, &props, 0));
}

TextureBackingMemoryCUDA::TextureBackingMemoryCUDA(TextureBackingMemoryCUDA&& other) noexcept
    : gpu(other.gpu)
    , size(other.size)
    , allocSize(other.allocSize)
    , memHandle(other.memHandle)
{
    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;
}

TextureBackingMemoryCUDA::~TextureBackingMemoryCUDA()
{
    if(allocSize != 0)
    {
        CUDA_DRIVER_CHECK(cuMemRelease(memHandle));
    }
}

TextureBackingMemoryCUDA& TextureBackingMemoryCUDA::operator=(TextureBackingMemoryCUDA&& other) noexcept
{
    if(allocSize != 0)
    {
        CUDA_DRIVER_CHECK(cuMemRelease(memHandle));
    }
    allocSize = other.allocSize;
    gpu = other.gpu;
    size = other.size;
    memHandle = other.memHandle;

    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;


    return *this;
}

void TextureBackingMemoryCUDA::ResizeBuffer(size_t newSize)
{
    TextureBackingMemoryCUDA newMem(*gpu, newSize);
    *this = TextureBackingMemoryCUDA(*gpu, newSize);
}

const GPUDeviceCUDA& TextureBackingMemoryCUDA::Device() const
{
    return *gpu;
}

size_t TextureBackingMemoryCUDA::Size() const
{
    return size;
}

}

template class mray::cuda::TextureCUDA<2, Float>;
template class mray::cuda::TextureCUDA<2, Vector2>;
template class mray::cuda::TextureCUDA<2, Vector3>;
template class mray::cuda::TextureCUDA<2, Vector4>;

template class mray::cuda::TextureCUDA<2, uint8_t>;
template class mray::cuda::TextureCUDA<2, Vector2uc>;
template class mray::cuda::TextureCUDA<2, Vector3uc>;
template class mray::cuda::TextureCUDA<2, Vector4uc>;

template class mray::cuda::TextureCUDA<2, int8_t>;
template class mray::cuda::TextureCUDA<2, Vector2c>;
template class mray::cuda::TextureCUDA<2, Vector3c>;
template class mray::cuda::TextureCUDA<2, Vector4c>;

template class mray::cuda::TextureCUDA<2, uint16_t>;
template class mray::cuda::TextureCUDA<2, Vector2us>;
template class mray::cuda::TextureCUDA<2, Vector3us>;
template class mray::cuda::TextureCUDA<2, Vector4us>;

template class mray::cuda::TextureCUDA<2, int16_t>;
template class mray::cuda::TextureCUDA<2, Vector2s>;
template class mray::cuda::TextureCUDA<2, Vector3s>;
template class mray::cuda::TextureCUDA<2, Vector4s>;