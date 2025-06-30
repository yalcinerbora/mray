#include "TextureCUDA.h"
#include "Core/GraphicsFunctions.h"

namespace mray::cuda
{

// Doing this instead of partial template classes etc
// There is only three dimensions so it is clearer?
template <uint32_t D, uint32_t I, class T>
cudaExtent MakeCudaExtent(const T& dim)
{
    if constexpr(D == 1)
    {
        return make_cudaExtent(dim, I, I);
    }
    else if constexpr(D == 2)
    {
        return make_cudaExtent(dim[0], dim[1], I);
    }
    else if constexpr(D == 3)
    {
        return make_cudaExtent(dim[0], dim[1], dim[2]);
    }
}

template <uint32_t D, uint32_t I, class T>
cudaPos MakeCudaPos(const T& dim)
{
    if constexpr(D == 1)
    {
        return make_cudaPos(dim, I, I);
    }
    else if constexpr(D == 2)
    {
        return make_cudaPos(dim[0], dim[1], I);
    }
    else if constexpr(D == 3)
    {
        return make_cudaPos(dim[0], dim[1], dim[2]);
    }
}

inline cudaTextureAddressMode DetermineAddressMode(MRayTextureEdgeResolveEnum e)
{
    switch(e)
    {
        case MRayTextureEdgeResolveEnum::MR_WRAP:
            return cudaTextureAddressMode::cudaAddressModeWrap;
        case MRayTextureEdgeResolveEnum::MR_CLAMP:
            return cudaTextureAddressMode::cudaAddressModeClamp;
        case MRayTextureEdgeResolveEnum::MR_MIRROR:
            return cudaTextureAddressMode::cudaAddressModeMirror;
        default:
            throw MRayError("Unknown edge resolve type for CUDA!");
    }
}

inline cudaTextureFilterMode DetermineFilterMode(MRayTextureInterpEnum i)
{
    switch(i)
    {
        case MRayTextureInterpEnum::MR_NEAREST:
            return cudaTextureFilterMode::cudaFilterModePoint;
        case MRayTextureInterpEnum::MR_LINEAR:
            return cudaTextureFilterMode::cudaFilterModeLinear;
        default:
            throw MRayError("Unknown texture interpolation type for CUDA!");
    }
}

template<uint32_t D, class T>
TextureCUDA_Normal<D, T>::TextureCUDA_Normal(const GPUDeviceCUDA& device,
                                             const TextureInitParams<D>& p)
    : gpu(&device)
    , tex(0)
    , texParams(p)
{
    // Warnings
    if(texParams.normIntegers && !IsNormConvertible)
    {
        MRAY_WARNING_LOG("Requested channel type cannot be converted to normalized form."
                         " Setting \"unormIntegers\" to false");
        texParams.normIntegers = false;
    };

    cudaExtent extent = MakeCudaExtent<D, 0u>(p.size);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<CudaType>();
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_MEM_THROW(cudaMallocMipmappedArray(&data, &channelDesc, extent, p.mipCount,
                                            cudaArrayDeferredMapping |
                                            cudaArraySurfaceLoadStore));

    cudaArrayMemoryRequirements memReq;
    CUDA_CHECK(cudaMipmappedArrayGetMemoryRequirements(&memReq, data, gpu->DeviceId()));
    alignment = memReq.alignment;
    size = memReq.size;

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    cudaTextureDesc tDesc = {};
    tDesc.addressMode[0] = DetermineAddressMode(texParams.eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(texParams.eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(texParams.eResolve);
    tDesc.filterMode = DetermineFilterMode(texParams.interp);
    tDesc.mipmapFilterMode = DetermineFilterMode(texParams.interp);
    tDesc.readMode = (texParams.normIntegers) ? cudaReadModeNormalizedFloat
                                              : cudaReadModeElementType;
    tDesc.sRGB = texParams.convertSRGB;
    // Border color can only be zero?
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = texParams.normCoordinates;

    tDesc.maxAnisotropy = texParams.maxAnisotropy;
    tDesc.mipmapLevelBias = texParams.mipmapBias;
    tDesc.minMipmapLevelClamp = texParams.minMipmapClamp;
    tDesc.maxMipmapLevelClamp = texParams.maxMipmapClamp;

    CUDA_CHECK(cudaCreateTextureObject(&tex, &rDesc, &tDesc, nullptr));

}

template<uint32_t D, class T>
TextureCUDA_Normal<D, T>::TextureCUDA_Normal(TextureCUDA_Normal&& other) noexcept
    : gpu(other.gpu)
    , tex(other.tex)
    , data(other.data)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{
    other.data = nullptr;
    other.tex = cudaTextureObject_t(0);
}

template<uint32_t D, class T>
TextureCUDA_Normal<D, T>& TextureCUDA_Normal<D, T>::operator=(TextureCUDA_Normal&& other) noexcept
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(tex));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    gpu = other.gpu;
    tex = other.tex;
    data = other.data;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;

    other.data = nullptr;
    other.tex = cudaTextureObject_t(0);
    return *this;
}

template<uint32_t D, class T>
TextureCUDA_Normal<D, T>::~TextureCUDA_Normal()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(tex));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }
}

template<uint32_t D, class T>
RWTextureRefCUDA<D, T> TextureCUDA_Normal<D, T>::GenerateRWRef(uint32_t mipLevel)
{
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Requested out of bounds mip level!");

    // TODO: Check if we are owning this cudaArray_t.
    // Since it is a "get" function we do not own this I guess
    cudaArray_t mipLevelArray = nullptr;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&mipLevelArray, data,
                                          static_cast<unsigned int>(mipLevel)));

    cudaSurfaceObject_t surf;
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = mipLevelArray;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &desc));

    return RWTextureRefCUDA<D, T>(surf);
}

template<uint32_t D, class T>
size_t TextureCUDA_Normal<D, T>::Size() const
{
    return size;
}

template<uint32_t D, class T>
size_t TextureCUDA_Normal<D, T>::Alignment() const
{
    return alignment;
}

template<uint32_t D, class T>
TextureExtent<D> TextureCUDA_Normal<D, T>::Extents() const
{
    return texParams.size;
}

template<uint32_t D, class T>
uint32_t TextureCUDA_Normal<D, T>::MipCount() const
{
    return texParams.mipCount;
}

template<uint32_t D, class T>
const GPUDeviceCUDA& TextureCUDA_Normal<D, T>::Device() const
{
    assert(gpu != nullptr);
    return *gpu;
}

template<uint32_t D, class T>
void TextureCUDA_Normal<D, T>::CommitMemory(const GPUQueueCUDA& queue,
                                            const TextureBackingMemoryCUDA& deviceMem,
                                            size_t offset)
{
    assert(deviceMem.Device() == *gpu);
    // Given span of memory, commit for usage.
    CUarrayMapInfo mappingInfo = {};
    mappingInfo.resourceType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mappingInfo.resource.mipmap = std::bit_cast<CUmipmappedArray>(data);

    mappingInfo.memHandleType = CU_MEM_HANDLE_TYPE_GENERIC;
    mappingInfo.memHandle.memHandle = ToHandleCUDA(deviceMem);

    mappingInfo.memOperationType = CU_MEM_OPERATION_TYPE_MAP;

    mappingInfo.offset = offset;
    mappingInfo.deviceBitMask = (1 << gpu->DeviceId());

    CUDA_DRIVER_CHECK(cuMemMapArrayAsync(&mappingInfo, 1, ToHandleCUDA(queue)));
}

template<uint32_t D, class T>
void TextureCUDA_Normal<D, T>::CopyFromAsync(const GPUQueueCUDA& queue,
                                             uint32_t mipLevel,
                                             const TextureExtent<D>& offset,
                                             const TextureExtent<D>& sizes,
                                             Span<const PaddedChannelType> regionFrom)
{
    cudaArray_t levelArray = nullptr;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyDefault;
    p.extent = MakeCudaExtent<D, 1>(sizes);

    p.dstArray = levelArray;
    p.dstPos = MakeCudaPos<D, 0>(offset);

    p.srcPos = make_cudaPos(0, 0, 0);
    //
    void* ptr = const_cast<Byte*>(reinterpret_cast<const Byte*>(regionFrom.data()));
    p.srcPtr = make_cudaPitchedPtr(ptr,
                                   p.extent.width * sizeof(PaddedChannelType),
                                   p.extent.width, p.extent.height);
    CUDA_CHECK(cudaMemcpy3DAsync(&p, ToHandleCUDA(queue)));
}

template<uint32_t D, class T>
void TextureCUDA_Normal<D, T>::CopyToAsync(Span<PaddedChannelType> regionFrom,
                                           const GPUQueueCUDA& queue,
                                           uint32_t mipLevel,
                                           const TextureExtent<D>& offset,
                                           const TextureExtent<D>& sizes)
{
    cudaArray_t levelArray = nullptr;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyDefault;
    p.extent = MakeCudaExtent<D, 1>(sizes);

    p.srcArray = levelArray;
    p.srcPos = MakeCudaPos<D, 0>(offset);
    //
    p.dstPos = make_cudaPos(0, 0, 0);
    p.dstPtr = make_cudaPitchedPtr(regionFrom.data(),
                                   p.extent.width * sizeof(PaddedChannelType),
                                   p.extent.width, p.extent.height);
    CUDA_CHECK(cudaMemcpy3DAsync(&p, ToHandleCUDA(queue)));
}

template<class T>
TextureCUDA_BC<T>::TextureCUDA_BC(const GPUDeviceCUDA& device,
                                  const TextureInitParams<2>& p)
    : gpu(&device)
    , tex(0)
    , texParams(p)
{
    // Warnings
    if(texParams.normIntegers && !IsNormConvertible)
    {
        MRAY_WARNING_LOG("Requested channel type cannot be converted to normalized form."
                         " Setting \"unormIntegers\" to false");
        texParams.normIntegers = false;
    };

    cudaExtent extent = MakeCudaExtent<2, 0u>(texParams.size);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<CudaTypeEnum>();
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_MEM_THROW(cudaMallocMipmappedArray(&data, &channelDesc, extent,
                                            texParams.mipCount,
                                            cudaArrayDeferredMapping));
    cudaArrayMemoryRequirements memReq;
    CUDA_CHECK(cudaMipmappedArrayGetMemoryRequirements(&memReq, data, gpu->DeviceId()));
    alignment = memReq.alignment;
    size = memReq.size;

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    cudaTextureDesc tDesc = {};
    tDesc.addressMode[0] = DetermineAddressMode(texParams.eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(texParams.eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(texParams.eResolve);
    tDesc.filterMode = DetermineFilterMode(texParams.interp);
    tDesc.mipmapFilterMode = DetermineFilterMode(texParams.interp);
    // Warnings
    bool isBC6 = (std::is_same_v<T, PixelBC6U> || std::is_same_v<T, PixelBC6S>);
    if(isBC6 && texParams.normIntegers == true)
    {
        MRAY_WARNING_LOG("BC6 textures must be read as \"unnormalized\"."
                         " Setting \"unormIntegers\" to false");
    }
    else if(!isBC6 && texParams.normIntegers == false)
    {
        MRAY_WARNING_LOG("Non-BC6 Block compressed textures must be read as \"normalized\"."
                         " Setting \"unormIntegers\" to true");
    }
    // BC6 requires element type
    // Other BC formats require normalized float
    // Which is "element mode" for these
    tDesc.readMode = (isBC6)
        ? cudaReadModeElementType
        : cudaReadModeNormalizedFloat;
    // Never utilize srgb conversion
    tDesc.sRGB = false;
    // Border color can only be zero?
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = texParams.normCoordinates;

    tDesc.maxAnisotropy = texParams.maxAnisotropy;
    tDesc.mipmapLevelBias = texParams.mipmapBias;
    tDesc.minMipmapLevelClamp = texParams.minMipmapClamp;
    tDesc.maxMipmapLevelClamp = texParams.maxMipmapClamp;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &rDesc, &tDesc, nullptr));
}

template<class T>
TextureCUDA_BC<T>::TextureCUDA_BC(TextureCUDA_BC&& other) noexcept
    : gpu(other.gpu)
    , tex(other.tex)
    , data(other.data)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{
    other.data = nullptr;
    other.tex = cudaTextureObject_t(0);
}

template<class T>
TextureCUDA_BC<T>& TextureCUDA_BC<T>::operator=(TextureCUDA_BC&& other) noexcept
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(tex));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    gpu = other.gpu;
    tex = other.tex;
    data = other.data;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;

    other.data = nullptr;
    other.tex = cudaTextureObject_t(0);
    return *this;
}

template<class T>
TextureCUDA_BC<T>::~TextureCUDA_BC()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(tex));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }
}

template<class T>
size_t TextureCUDA_BC<T>::Size() const
{
    return size;
}

template<class T>
size_t TextureCUDA_BC<T>::Alignment() const

{
    return alignment;
}

template<class T>
TextureExtent<2> TextureCUDA_BC<T>::Extents() const
{
    return texParams.size;
}

template<class T>
uint32_t TextureCUDA_BC<T>::MipCount() const
{
    return texParams.mipCount;
}

template<class T>
const GPUDeviceCUDA& TextureCUDA_BC<T>::Device() const
{
    assert(gpu != nullptr);
    return *gpu;
}

template<class T>
void TextureCUDA_BC<T>::CommitMemory(const GPUQueueCUDA& queue,
                                     const TextureBackingMemoryCUDA& deviceMem,
                                     size_t offset)
{
    assert(deviceMem.Device() == *gpu);
    // Given span of memory, commit for usage.
    CUarrayMapInfo mappingInfo = {};
    mappingInfo.resourceType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mappingInfo.resource.mipmap = std::bit_cast<CUmipmappedArray>(data);

    mappingInfo.memHandleType = CU_MEM_HANDLE_TYPE_GENERIC;
    mappingInfo.memHandle.memHandle = ToHandleCUDA(deviceMem);

    mappingInfo.memOperationType = CU_MEM_OPERATION_TYPE_MAP;

    mappingInfo.offset = offset;
    mappingInfo.deviceBitMask = (1 << gpu->DeviceId());

    CUDA_DRIVER_CHECK(cuMemMapArrayAsync(&mappingInfo, 1, ToHandleCUDA(queue)));
}

template<class T>
void TextureCUDA_BC<T>::CopyFromAsync(const GPUQueueCUDA& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<2>& offset,
                                      const TextureExtent<2>& sizes,
                                      Span<const PaddedChannelType> regionFrom)
{
    // Silently ignore the mip if out of bounds
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Out of bounds mip level is requested!");

    cudaArray_t levelArray = nullptr;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));
    void* ptr = const_cast<Byte*>(reinterpret_cast<const Byte*>(regionFrom.data()));
    auto srcWH = Math::DivideUp(sizes, Vector2ui(BC_TILE_SIZE));
    auto dstWH = Math::NextMultiple(sizes, Vector2ui(BC_TILE_SIZE));
    size_t srcPitch = srcWH[0] * sizeof(PaddedChannelType);

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyDefault;
    p.extent = MakeCudaExtent<2, 1>(dstWH);
    p.dstArray = levelArray;
    p.dstPos = MakeCudaPos<2, 0>(offset);
    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(ptr, srcPitch, srcWH[0], srcWH[1]);
    CUDA_CHECK(cudaMemcpy3DAsync(&p, ToHandleCUDA(queue)));
}

template<class T>
void TextureCUDA_BC<T>::CopyToAsync(Span<PaddedChannelType> regionFrom,
                                    const GPUQueueCUDA& queue,
                                    uint32_t mipLevel,
                                    const TextureExtent<2>& offset,
                                    const TextureExtent<2>& sizes) const
{
    // Silently ignore the mip if out of bounds
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Out of bounds mip level is requested!");

    cudaArray_t levelArray = nullptr;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));
    void* ptr = reinterpret_cast<Byte*>(regionFrom.data());
    auto srcWH = Math::NextMultiple(sizes, Vector2ui(BC_TILE_SIZE));
    auto dstWH = Math::DivideUp(sizes, Vector2ui(BC_TILE_SIZE));
    size_t dstPitch = dstWH[0] * sizeof(PaddedChannelType);

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyDefault;
    p.extent = MakeCudaExtent<2, 1>(srcWH);
    p.srcArray = levelArray;
    p.srcPos = MakeCudaPos<2, 0>(offset);
    p.dstPos = make_cudaPos(0, 0, 0);
    p.dstPtr = make_cudaPitchedPtr(ptr, dstPitch, dstWH[0], dstWH[1]);
    CUDA_CHECK(cudaMemcpy3DAsync(&p, ToHandleCUDA(queue)));
}

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
    allocSize = Math::NextMultiple(size, granularity);
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

// Common Textures 1D
template class mray::cuda::TextureCUDA_Normal<1, Float>;
template class mray::cuda::TextureCUDA_Normal<1, Vector2>;
template class mray::cuda::TextureCUDA_Normal<1, Vector3>;
template class mray::cuda::TextureCUDA_Normal<1, Vector4>;

template class mray::cuda::TextureCUDA_Normal<1, uint8_t>;
template class mray::cuda::TextureCUDA_Normal<1, Vector2uc>;
template class mray::cuda::TextureCUDA_Normal<1, Vector3uc>;
template class mray::cuda::TextureCUDA_Normal<1, Vector4uc>;

template class mray::cuda::TextureCUDA_Normal<1, int8_t>;
template class mray::cuda::TextureCUDA_Normal<1, Vector2c>;
template class mray::cuda::TextureCUDA_Normal<1, Vector3c>;
template class mray::cuda::TextureCUDA_Normal<1, Vector4c>;

template class mray::cuda::TextureCUDA_Normal<1, uint16_t>;
template class mray::cuda::TextureCUDA_Normal<1, Vector2us>;
template class mray::cuda::TextureCUDA_Normal<1, Vector3us>;
template class mray::cuda::TextureCUDA_Normal<1, Vector4us>;

template class mray::cuda::TextureCUDA_Normal<1, int16_t>;
template class mray::cuda::TextureCUDA_Normal<1, Vector2s>;
template class mray::cuda::TextureCUDA_Normal<1, Vector3s>;
template class mray::cuda::TextureCUDA_Normal<1, Vector4s>;

// Common Textures 2D
template class mray::cuda::TextureCUDA_Normal<2, Float>;
template class mray::cuda::TextureCUDA_Normal<2, Vector2>;
template class mray::cuda::TextureCUDA_Normal<2, Vector3>;
template class mray::cuda::TextureCUDA_Normal<2, Vector4>;

template class mray::cuda::TextureCUDA_Normal<2, uint8_t>;
template class mray::cuda::TextureCUDA_Normal<2, Vector2uc>;
template class mray::cuda::TextureCUDA_Normal<2, Vector3uc>;
template class mray::cuda::TextureCUDA_Normal<2, Vector4uc>;

template class mray::cuda::TextureCUDA_Normal<2, int8_t>;
template class mray::cuda::TextureCUDA_Normal<2, Vector2c>;
template class mray::cuda::TextureCUDA_Normal<2, Vector3c>;
template class mray::cuda::TextureCUDA_Normal<2, Vector4c>;

template class mray::cuda::TextureCUDA_Normal<2, uint16_t>;
template class mray::cuda::TextureCUDA_Normal<2, Vector2us>;
template class mray::cuda::TextureCUDA_Normal<2, Vector3us>;
template class mray::cuda::TextureCUDA_Normal<2, Vector4us>;

template class mray::cuda::TextureCUDA_Normal<2, int16_t>;
template class mray::cuda::TextureCUDA_Normal<2, Vector2s>;
template class mray::cuda::TextureCUDA_Normal<2, Vector3s>;
template class mray::cuda::TextureCUDA_Normal<2, Vector4s>;

template class mray::cuda::TextureCUDA_BC<PixelBC1>;
template class mray::cuda::TextureCUDA_BC<PixelBC2>;
template class mray::cuda::TextureCUDA_BC<PixelBC3>;
template class mray::cuda::TextureCUDA_BC<PixelBC4U>;
template class mray::cuda::TextureCUDA_BC<PixelBC4S>;
template class mray::cuda::TextureCUDA_BC<PixelBC5U>;
template class mray::cuda::TextureCUDA_BC<PixelBC5S>;
template class mray::cuda::TextureCUDA_BC<PixelBC6U>;
template class mray::cuda::TextureCUDA_BC<PixelBC6S>;
template class mray::cuda::TextureCUDA_BC<PixelBC7>;

// Common Textures 3D
template class mray::cuda::TextureCUDA_Normal<3, Float>;
template class mray::cuda::TextureCUDA_Normal<3, Vector2>;
template class mray::cuda::TextureCUDA_Normal<3, Vector3>;
template class mray::cuda::TextureCUDA_Normal<3, Vector4>;

template class mray::cuda::TextureCUDA_Normal<3, uint8_t>;
template class mray::cuda::TextureCUDA_Normal<3, Vector2uc>;
template class mray::cuda::TextureCUDA_Normal<3, Vector3uc>;
template class mray::cuda::TextureCUDA_Normal<3, Vector4uc>;

template class mray::cuda::TextureCUDA_Normal<3, int8_t>;
template class mray::cuda::TextureCUDA_Normal<3, Vector2c>;
template class mray::cuda::TextureCUDA_Normal<3, Vector3c>;
template class mray::cuda::TextureCUDA_Normal<3, Vector4c>;

template class mray::cuda::TextureCUDA_Normal<3, uint16_t>;
template class mray::cuda::TextureCUDA_Normal<3, Vector2us>;
template class mray::cuda::TextureCUDA_Normal<3, Vector3us>;
template class mray::cuda::TextureCUDA_Normal<3, Vector4us>;

template class mray::cuda::TextureCUDA_Normal<3, int16_t>;
template class mray::cuda::TextureCUDA_Normal<3, Vector2s>;
template class mray::cuda::TextureCUDA_Normal<3, Vector3s>;
template class mray::cuda::TextureCUDA_Normal<3, Vector4s>;

// Common Textures 1D
template class mray::cuda::TextureCUDA<1, Float>;
template class mray::cuda::TextureCUDA<1, Vector2>;
template class mray::cuda::TextureCUDA<1, Vector3>;
template class mray::cuda::TextureCUDA<1, Vector4>;

template class mray::cuda::TextureCUDA<1, uint8_t>;
template class mray::cuda::TextureCUDA<1, Vector2uc>;
template class mray::cuda::TextureCUDA<1, Vector3uc>;
template class mray::cuda::TextureCUDA<1, Vector4uc>;

template class mray::cuda::TextureCUDA<1, int8_t>;
template class mray::cuda::TextureCUDA<1, Vector2c>;
template class mray::cuda::TextureCUDA<1, Vector3c>;
template class mray::cuda::TextureCUDA<1, Vector4c>;

template class mray::cuda::TextureCUDA<1, uint16_t>;
template class mray::cuda::TextureCUDA<1, Vector2us>;
template class mray::cuda::TextureCUDA<1, Vector3us>;
template class mray::cuda::TextureCUDA<1, Vector4us>;

template class mray::cuda::TextureCUDA<1, int16_t>;
template class mray::cuda::TextureCUDA<1, Vector2s>;
template class mray::cuda::TextureCUDA<1, Vector3s>;
template class mray::cuda::TextureCUDA<1, Vector4s>;

// Common Textures 2D
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

template class mray::cuda::TextureCUDA<2, PixelBC1>;
template class mray::cuda::TextureCUDA<2, PixelBC2>;
template class mray::cuda::TextureCUDA<2, PixelBC3>;
template class mray::cuda::TextureCUDA<2, PixelBC4U>;
template class mray::cuda::TextureCUDA<2, PixelBC4S>;
template class mray::cuda::TextureCUDA<2, PixelBC5U>;
template class mray::cuda::TextureCUDA<2, PixelBC5S>;
template class mray::cuda::TextureCUDA<2, PixelBC6U>;
template class mray::cuda::TextureCUDA<2, PixelBC6S>;
template class mray::cuda::TextureCUDA<2, PixelBC7>;

// Common Textures 3D
template class mray::cuda::TextureCUDA<3, Float>;
template class mray::cuda::TextureCUDA<3, Vector2>;
template class mray::cuda::TextureCUDA<3, Vector3>;
template class mray::cuda::TextureCUDA<3, Vector4>;

template class mray::cuda::TextureCUDA<3, uint8_t>;
template class mray::cuda::TextureCUDA<3, Vector2uc>;
template class mray::cuda::TextureCUDA<3, Vector3uc>;
template class mray::cuda::TextureCUDA<3, Vector4uc>;

template class mray::cuda::TextureCUDA<3, int8_t>;
template class mray::cuda::TextureCUDA<3, Vector2c>;
template class mray::cuda::TextureCUDA<3, Vector3c>;
template class mray::cuda::TextureCUDA<3, Vector4c>;

template class mray::cuda::TextureCUDA<3, uint16_t>;
template class mray::cuda::TextureCUDA<3, Vector2us>;
template class mray::cuda::TextureCUDA<3, Vector3us>;
template class mray::cuda::TextureCUDA<3, Vector4us>;

template class mray::cuda::TextureCUDA<3, int16_t>;
template class mray::cuda::TextureCUDA<3, Vector2s>;
template class mray::cuda::TextureCUDA<3, Vector3s>;
template class mray::cuda::TextureCUDA<3, Vector4s>;