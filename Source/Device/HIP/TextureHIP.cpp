#include "TextureHIP.h"
#include "Core/GraphicsFunctions.h"

namespace mray::hip
{

// Doing this instead of partial template classes etc
// There is only three dimensions so it is clearer?
template <uint32_t D, uint32_t I, class T>
hipExtent MakeHipExtent(const T& dim)
{
    if constexpr(D == 1)
    {
        return make_hipExtent(dim, I, I);
    }
    else if constexpr(D == 2)
    {
        return make_hipExtent(dim[0], dim[1], I);
    }
    else if constexpr(D == 3)
    {
        return make_hipExtent(dim[0], dim[1], dim[2]);
    }
}

template <uint32_t D, uint32_t I, class T>
hipPos MakeHipPos(const T& dim)
{
    if constexpr(D == 1)
    {
        return make_hipPos(dim, I, I);
    }
    else if constexpr(D == 2)
    {
        return make_hipPos(dim[0], dim[1], I);
    }
    else if constexpr(D == 3)
    {
        return make_hipPos(dim[0], dim[1], dim[2]);
    }
}

inline hipTextureAddressMode DetermineAddressMode(MRayTextureEdgeResolveEnum e)
{
    switch(e)
    {
        case MRayTextureEdgeResolveEnum::MR_WRAP:
            return hipTextureAddressMode::hipAddressModeWrap;
        case MRayTextureEdgeResolveEnum::MR_CLAMP:
            return hipTextureAddressMode::hipAddressModeClamp;
        case MRayTextureEdgeResolveEnum::MR_MIRROR:
            return hipTextureAddressMode::hipAddressModeMirror;
        default:
            throw MRayError("Unknown edge resolve type for HIP!");
    }
}

inline hipTextureFilterMode DetermineFilterMode(MRayTextureInterpEnum i)
{
    switch(i)
    {
        case MRayTextureInterpEnum::MR_NEAREST:
            return hipTextureFilterMode::hipFilterModePoint;
        case MRayTextureInterpEnum::MR_LINEAR:
            return hipTextureFilterMode::hipFilterModeLinear;
        default:
            throw MRayError("Unknown texture interpolation type for HIP!");
    }
}

template<uint32_t D, class T>
TextureHIP_Normal<D, T>::TextureHIP_Normal(const GPUDeviceHIP& device,
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

    hipExtent extent = MakeHipExtent<D, 0u>(p.size);
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<HipType>();
    HIP_CHECK(hipSetDevice(gpu->DeviceId()));
    HIP_MEM_THROW(hipMallocMipmappedArray(&data, &channelDesc, extent, p.mipCount,
                                          // TODO: HIP do not have deferred mapping support
                                          // Change this later
                                          //hipArrayDeferredMapping |
                                          hipArraySurfaceLoadStore));
    // TODO: Again hip do not support this
    // hipArrayMemoryRequirements memReq;
    // HIP_CHECK(hipMipmappedArrayGetMemoryRequirements(&memReq, data, gpu->DeviceId()));
    // alignment = memReq.alignment;
    // size = memReq.size;

    // Allocation Done now generate texture
    hipResourceDesc rDesc = {};
    rDesc.resType = hipResourceType::hipResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    hipTextureDesc tDesc = {};
    tDesc.addressMode[0] = DetermineAddressMode(texParams.eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(texParams.eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(texParams.eResolve);
    tDesc.filterMode = DetermineFilterMode(texParams.interp);
    tDesc.mipmapFilterMode = DetermineFilterMode(texParams.interp);
    tDesc.readMode = (texParams.normIntegers) ? hipReadModeNormalizedFloat
                                              : hipReadModeElementType;
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

    HIP_CHECK(hipCreateTextureObject(&tex, &rDesc, &tDesc, nullptr));

}

template<uint32_t D, class T>
TextureHIP_Normal<D, T>::TextureHIP_Normal(TextureHIP_Normal&& other) noexcept
    : gpu(other.gpu)
    , tex(other.tex)
    , data(other.data)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{
    other.data = nullptr;
    other.tex = hipTextureObject_t(0);
}

template<uint32_t D, class T>
TextureHIP_Normal<D, T>& TextureHIP_Normal<D, T>::operator=(TextureHIP_Normal&& other) noexcept
{
    assert(this != &other);
    if(data)
    {
        HIP_CHECK(hipSetDevice(gpu->DeviceId()));
        HIP_CHECK(hipDestroyTextureObject(tex));
        HIP_CHECK(hipFreeMipmappedArray(data));
    }

    gpu = other.gpu;
    tex = other.tex;
    data = other.data;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;

    other.data = nullptr;
    other.tex = hipTextureObject_t(0);
    return *this;
}

template<uint32_t D, class T>
TextureHIP_Normal<D, T>::~TextureHIP_Normal()
{
    if(data)
    {
        HIP_CHECK(hipSetDevice(gpu->DeviceId()));
        HIP_CHECK(hipDestroyTextureObject(tex));
        HIP_CHECK(hipFreeMipmappedArray(data));
    }
}

template<uint32_t D, class T>
RWTextureRefHIP<D, T> TextureHIP_Normal<D, T>::GenerateRWRef(uint32_t mipLevel)
{
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Requested out of bounds mip level!");

    // TODO: Check if we are owning this hipArray_t.
    // Since it is a "get" function we do not own this I guess
    hipArray_t mipLevelArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&mipLevelArray, data,
                                        static_cast<unsigned int>(mipLevel)));

    hipSurfaceObject_t surf;
    hipResourceDesc desc = {};
    desc.resType = hipResourceTypeArray;
    desc.res.array.array = mipLevelArray;
    HIP_CHECK(hipCreateSurfaceObject(&surf, &desc));

    return RWTextureRefHIP<D, T>(surf);
}

template<uint32_t D, class T>
size_t TextureHIP_Normal<D, T>::Size() const
{
    return size;
}

template<uint32_t D, class T>
size_t TextureHIP_Normal<D, T>::Alignment() const
{
    return alignment;
}

template<uint32_t D, class T>
TextureExtent<D> TextureHIP_Normal<D, T>::Extents() const
{
    return texParams.size;
}

template<uint32_t D, class T>
uint32_t TextureHIP_Normal<D, T>::MipCount() const
{
    return texParams.mipCount;
}

template<uint32_t D, class T>
const GPUDeviceHIP& TextureHIP_Normal<D, T>::Device() const
{
    assert(gpu != nullptr);
    return *gpu;
}

template<uint32_t D, class T>
void TextureHIP_Normal<D, T>::CommitMemory(const GPUQueueHIP& queue,
                                            const TextureBackingMemoryHIP& deviceMem,
                                            size_t offset)
{
    assert(deviceMem.Device() == *gpu);
    // Given span of memory, commit for usage.
    hipArrayMapInfo mappingInfo = {};
    mappingInfo.resourceType = hipResourceTypeMipmappedArray;
    mappingInfo.resource.mipmap = *data;

    mappingInfo.memHandleType = hipMemHandleTypeGeneric;
    mappingInfo.memHandle.memHandle = ToHandleHIP(deviceMem);

    mappingInfo.memOperationType = hipMemOperationTypeMap;

    mappingInfo.offset = offset;
    mappingInfo.deviceBitMask = (1 << gpu->DeviceId());

    HIP_DRIVER_CHECK(hipMemMapArrayAsync(&mappingInfo, 1, ToHandleHIP(queue)));
}

template<uint32_t D, class T>
void TextureHIP_Normal<D, T>::CopyFromAsync(const GPUQueueHIP& queue,
                                            uint32_t mipLevel,
                                            const TextureExtent<D>& offset,
                                            const TextureExtent<D>& sizes,
                                            Span<const PaddedChannelType> regionFrom)
{
    hipArray_t levelArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    hipMemcpy3DParms p = {};
    p.kind = hipMemcpyDefault;
    p.extent = MakeHipExtent<D, 1>(sizes);

    p.dstArray = levelArray;
    p.dstPos = MakeHipPos<D, 0>(offset);

    p.srcPos = make_hipPos(0, 0, 0);
    //
    void* ptr = const_cast<Byte*>(reinterpret_cast<const Byte*>(regionFrom.data()));
    p.srcPtr = make_hipPitchedPtr(ptr,
                                  p.extent.width * sizeof(PaddedChannelType),
                                  p.extent.width, p.extent.height);
    HIP_CHECK(hipMemcpy3DAsync(&p, ToHandleHIP(queue)));
}

template<class T>
TextureHIP_BC<T>::TextureHIP_BC(const GPUDeviceHIP& device,
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

    hipExtent extent = MakeHipExtent<2, 0u>(texParams.size);
    // TODO: BC is not supported in HIP yet, so no enum-backed "hipCreateChannelDesc"
    // instantiation
    hipChannelFormatDesc channelDesc = {};// = hipCreateChannelDesc<HipTypeEnum>();
    HIP_CHECK(hipSetDevice(gpu->DeviceId()));
    HIP_MEM_THROW(hipMallocMipmappedArray(&data, &channelDesc, extent,
                                            texParams.mipCount
                                            //,
                                            // TODO: HIP do not have deferred mapping support
                                            // Change this later
                                            //hipArrayDeferredMapping
                                            ));

    // TODO: Again hip do not support this
    // hipArrayMemoryRequirements memReq;
    // HIP_CHECK(hipMipmappedArrayGetMemoryRequirements(&memReq, data, gpu->DeviceId()));
    // alignment = memReq.alignment;
    // size = memReq.size;

    // Allocation Done now generate texture
    hipResourceDesc rDesc = {};
    rDesc.resType = hipResourceType::hipResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    hipTextureDesc tDesc = {};
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
        ? hipReadModeElementType
        : hipReadModeNormalizedFloat;
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
    HIP_CHECK(hipCreateTextureObject(&tex, &rDesc, &tDesc, nullptr));
}

template<class T>
TextureHIP_BC<T>::TextureHIP_BC(TextureHIP_BC&& other) noexcept
    : gpu(other.gpu)
    , tex(other.tex)
    , data(other.data)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{
    other.data = nullptr;
    other.tex = hipTextureObject_t(0);
}

template<class T>
TextureHIP_BC<T>& TextureHIP_BC<T>::operator=(TextureHIP_BC&& other) noexcept
{
    assert(this != &other);
    if(data)
    {
        HIP_CHECK(hipSetDevice(gpu->DeviceId()));
        HIP_CHECK(hipDestroyTextureObject(tex));
        HIP_CHECK(hipFreeMipmappedArray(data));
    }

    gpu = other.gpu;
    tex = other.tex;
    data = other.data;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;

    other.data = nullptr;
    other.tex = hipTextureObject_t(0);
    return *this;
}

template<class T>
TextureHIP_BC<T>::~TextureHIP_BC()
{
    if(data)
    {
        HIP_CHECK(hipSetDevice(gpu->DeviceId()));
        HIP_CHECK(hipDestroyTextureObject(tex));
        HIP_CHECK(hipFreeMipmappedArray(data));
    }
}

template<class T>
size_t TextureHIP_BC<T>::Size() const
{
    return size;
}

template<class T>
size_t TextureHIP_BC<T>::Alignment() const

{
    return alignment;
}

template<class T>
TextureExtent<2> TextureHIP_BC<T>::Extents() const
{
    return texParams.size;
}

template<class T>
uint32_t TextureHIP_BC<T>::MipCount() const
{
    return texParams.mipCount;
}

template<class T>
const GPUDeviceHIP& TextureHIP_BC<T>::Device() const
{
    assert(gpu != nullptr);
    return *gpu;
}

template<class T>
void TextureHIP_BC<T>::CommitMemory(const GPUQueueHIP& queue,
                                    const TextureBackingMemoryHIP& deviceMem,
                                    size_t offset)
{
    assert(deviceMem.Device() == *gpu);
    // Given span of memory, commit for usage.
    hipArrayMapInfo mappingInfo = {};
    mappingInfo.resourceType = hipResourceTypeMipmappedArray;
    mappingInfo.resource.mipmap = *data;

    mappingInfo.memHandleType = hipMemHandleTypeGeneric;
    mappingInfo.memHandle.memHandle = ToHandleHIP(deviceMem);

    mappingInfo.memOperationType = hipMemOperationTypeMap;

    mappingInfo.offset = offset;
    mappingInfo.deviceBitMask = (1 << gpu->DeviceId());

    HIP_DRIVER_CHECK(hipMemMapArrayAsync(&mappingInfo, 1, ToHandleHIP(queue)));
}

template<class T>
void TextureHIP_BC<T>::CopyFromAsync(const GPUQueueHIP& queue,
                                     uint32_t mipLevel,
                                     const TextureExtent<2>& offset,
                                     const TextureExtent<2>& sizes,
                                     Span<const PaddedChannelType> regionFrom)
{
    // Silently ignore the mip if out of bounds
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Out of bounds mip level is requested!");

    hipArray_t levelArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&levelArray, data, mipLevel));
    void* ptr = const_cast<Byte*>(reinterpret_cast<const Byte*>(regionFrom.data()));
    auto srcWH = Math::DivideUp(sizes, Vector2ui(BC_TILE_SIZE));
    auto dstWH = Math::NextMultiple(sizes, Vector2ui(BC_TILE_SIZE));
    size_t srcPitch = srcWH[0] * sizeof(PaddedChannelType);

    hipMemcpy3DParms p = {};
    p.kind = hipMemcpyDefault;
    p.extent = MakeHipExtent<2, 1>(dstWH);
    p.dstArray = levelArray;
    p.dstPos = MakeHipPos<2, 0>(offset);
    p.srcPos = make_hipPos(0, 0, 0);
    p.srcPtr = make_hipPitchedPtr(ptr, srcPitch, srcWH[0], srcWH[1]);
    HIP_CHECK(hipMemcpy3DAsync(&p, ToHandleHIP(queue)));
}

template<class T>
void TextureHIP_BC<T>::CopyToAsync(Span<PaddedChannelType> regionFrom,
                                   const GPUQueueHIP& queue,
                                   uint32_t mipLevel,
                                   const TextureExtent<2>& offset,
                                   const TextureExtent<2>& sizes) const
{
    // Silently ignore the mip if out of bounds
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Out of bounds mip level is requested!");

    hipArray_t levelArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&levelArray, data, mipLevel));
    void* ptr = reinterpret_cast<Byte*>(regionFrom.data());
    auto srcWH = Math::NextMultiple(sizes, Vector2ui(BC_TILE_SIZE));
    auto dstWH = Math::DivideUp(sizes, Vector2ui(BC_TILE_SIZE));
    size_t dstPitch = dstWH[0] * sizeof(PaddedChannelType);

    hipMemcpy3DParms p = {};
    p.kind = hipMemcpyDefault;
    p.extent = MakeHipExtent<2, 1>(srcWH);
    p.srcArray = levelArray;
    p.srcPos = MakeHipPos<2, 0>(offset);
    p.dstPos = make_hipPos(0, 0, 0);
    p.dstPtr = make_hipPitchedPtr(ptr, dstPitch, dstWH[0], dstWH[1]);
    HIP_CHECK(hipMemcpy3DAsync(&p, ToHandleHIP(queue)));
}

TextureBackingMemoryHIP::TextureBackingMemoryHIP(const GPUDeviceHIP& gpu)
    : gpu(&gpu)
    , size(0)
    , allocSize(0)
    , memHandle(0)
{}

TextureBackingMemoryHIP::TextureBackingMemoryHIP(const GPUDeviceHIP& device, size_t sizeInBytes)
    : TextureBackingMemoryHIP(device)
{
    assert(sizeInBytes != 0);
    size = sizeInBytes;

    hipMemAllocationProp props = {};
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = gpu->DeviceId();
    props.type = hipMemAllocationTypePinned;
    // TODO: CU_MEM_CREATE_USAGE_TILE_POOL does not exist in HIP
    // zero is ok?????
    props.allocFlags.usage = 0;

    size_t granularity;
    HIP_DRIVER_CHECK(hipMemGetAllocationGranularity(&granularity, &props,
                                                    hipMemAllocationGranularityRecommended));
    allocSize = Math::NextMultiple(size, granularity);
    HIP_DRIVER_CHECK(hipMemCreate(&memHandle, allocSize, &props, 0));
}

TextureBackingMemoryHIP::TextureBackingMemoryHIP(TextureBackingMemoryHIP&& other) noexcept
    : gpu(other.gpu)
    , size(other.size)
    , allocSize(other.allocSize)
    , memHandle(other.memHandle)
{
    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;
}

TextureBackingMemoryHIP::~TextureBackingMemoryHIP()
{
    if(allocSize != 0)
    {
        HIP_DRIVER_CHECK(hipMemRelease(memHandle));
    }
}

TextureBackingMemoryHIP& TextureBackingMemoryHIP::operator=(TextureBackingMemoryHIP&& other) noexcept
{
    if(allocSize != 0)
    {
        HIP_DRIVER_CHECK(hipMemRelease(memHandle));
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

void TextureBackingMemoryHIP::ResizeBuffer(size_t newSize)
{
    TextureBackingMemoryHIP newMem(*gpu, newSize);
    *this = TextureBackingMemoryHIP(*gpu, newSize);
}

const GPUDeviceHIP& TextureBackingMemoryHIP::Device() const
{
    return *gpu;
}

size_t TextureBackingMemoryHIP::Size() const
{
    return size;
}

}

// Common Textures 1D
template class mray::hip::TextureHIP_Normal<1, Float>;
template class mray::hip::TextureHIP_Normal<1, Vector2>;
template class mray::hip::TextureHIP_Normal<1, Vector3>;
template class mray::hip::TextureHIP_Normal<1, Vector4>;

template class mray::hip::TextureHIP_Normal<1, uint8_t>;
template class mray::hip::TextureHIP_Normal<1, Vector2uc>;
template class mray::hip::TextureHIP_Normal<1, Vector3uc>;
template class mray::hip::TextureHIP_Normal<1, Vector4uc>;

template class mray::hip::TextureHIP_Normal<1, int8_t>;
template class mray::hip::TextureHIP_Normal<1, Vector2c>;
template class mray::hip::TextureHIP_Normal<1, Vector3c>;
template class mray::hip::TextureHIP_Normal<1, Vector4c>;

template class mray::hip::TextureHIP_Normal<1, uint16_t>;
template class mray::hip::TextureHIP_Normal<1, Vector2us>;
template class mray::hip::TextureHIP_Normal<1, Vector3us>;
template class mray::hip::TextureHIP_Normal<1, Vector4us>;

template class mray::hip::TextureHIP_Normal<1, int16_t>;
template class mray::hip::TextureHIP_Normal<1, Vector2s>;
template class mray::hip::TextureHIP_Normal<1, Vector3s>;
template class mray::hip::TextureHIP_Normal<1, Vector4s>;

// Common Textures 2D
template class mray::hip::TextureHIP_Normal<2, Float>;
template class mray::hip::TextureHIP_Normal<2, Vector2>;
template class mray::hip::TextureHIP_Normal<2, Vector3>;
template class mray::hip::TextureHIP_Normal<2, Vector4>;

template class mray::hip::TextureHIP_Normal<2, uint8_t>;
template class mray::hip::TextureHIP_Normal<2, Vector2uc>;
template class mray::hip::TextureHIP_Normal<2, Vector3uc>;
template class mray::hip::TextureHIP_Normal<2, Vector4uc>;

template class mray::hip::TextureHIP_Normal<2, int8_t>;
template class mray::hip::TextureHIP_Normal<2, Vector2c>;
template class mray::hip::TextureHIP_Normal<2, Vector3c>;
template class mray::hip::TextureHIP_Normal<2, Vector4c>;

template class mray::hip::TextureHIP_Normal<2, uint16_t>;
template class mray::hip::TextureHIP_Normal<2, Vector2us>;
template class mray::hip::TextureHIP_Normal<2, Vector3us>;
template class mray::hip::TextureHIP_Normal<2, Vector4us>;

template class mray::hip::TextureHIP_Normal<2, int16_t>;
template class mray::hip::TextureHIP_Normal<2, Vector2s>;
template class mray::hip::TextureHIP_Normal<2, Vector3s>;
template class mray::hip::TextureHIP_Normal<2, Vector4s>;

template class mray::hip::TextureHIP_BC<PixelBC1>;
template class mray::hip::TextureHIP_BC<PixelBC2>;
template class mray::hip::TextureHIP_BC<PixelBC3>;
template class mray::hip::TextureHIP_BC<PixelBC4U>;
template class mray::hip::TextureHIP_BC<PixelBC4S>;
template class mray::hip::TextureHIP_BC<PixelBC5U>;
template class mray::hip::TextureHIP_BC<PixelBC5S>;
template class mray::hip::TextureHIP_BC<PixelBC6U>;
template class mray::hip::TextureHIP_BC<PixelBC6S>;
template class mray::hip::TextureHIP_BC<PixelBC7>;

// Common Textures 3D
template class mray::hip::TextureHIP_Normal<3, Float>;
template class mray::hip::TextureHIP_Normal<3, Vector2>;
template class mray::hip::TextureHIP_Normal<3, Vector3>;
template class mray::hip::TextureHIP_Normal<3, Vector4>;

template class mray::hip::TextureHIP_Normal<3, uint8_t>;
template class mray::hip::TextureHIP_Normal<3, Vector2uc>;
template class mray::hip::TextureHIP_Normal<3, Vector3uc>;
template class mray::hip::TextureHIP_Normal<3, Vector4uc>;

template class mray::hip::TextureHIP_Normal<3, int8_t>;
template class mray::hip::TextureHIP_Normal<3, Vector2c>;
template class mray::hip::TextureHIP_Normal<3, Vector3c>;
template class mray::hip::TextureHIP_Normal<3, Vector4c>;

template class mray::hip::TextureHIP_Normal<3, uint16_t>;
template class mray::hip::TextureHIP_Normal<3, Vector2us>;
template class mray::hip::TextureHIP_Normal<3, Vector3us>;
template class mray::hip::TextureHIP_Normal<3, Vector4us>;

template class mray::hip::TextureHIP_Normal<3, int16_t>;
template class mray::hip::TextureHIP_Normal<3, Vector2s>;
template class mray::hip::TextureHIP_Normal<3, Vector3s>;
template class mray::hip::TextureHIP_Normal<3, Vector4s>;

// Common Textures 1D
template class mray::hip::TextureHIP<1, Float>;
template class mray::hip::TextureHIP<1, Vector2>;
template class mray::hip::TextureHIP<1, Vector3>;
template class mray::hip::TextureHIP<1, Vector4>;

template class mray::hip::TextureHIP<1, uint8_t>;
template class mray::hip::TextureHIP<1, Vector2uc>;
template class mray::hip::TextureHIP<1, Vector3uc>;
template class mray::hip::TextureHIP<1, Vector4uc>;

template class mray::hip::TextureHIP<1, int8_t>;
template class mray::hip::TextureHIP<1, Vector2c>;
template class mray::hip::TextureHIP<1, Vector3c>;
template class mray::hip::TextureHIP<1, Vector4c>;

template class mray::hip::TextureHIP<1, uint16_t>;
template class mray::hip::TextureHIP<1, Vector2us>;
template class mray::hip::TextureHIP<1, Vector3us>;
template class mray::hip::TextureHIP<1, Vector4us>;

template class mray::hip::TextureHIP<1, int16_t>;
template class mray::hip::TextureHIP<1, Vector2s>;
template class mray::hip::TextureHIP<1, Vector3s>;
template class mray::hip::TextureHIP<1, Vector4s>;

// Common Textures 2D
template class mray::hip::TextureHIP<2, Float>;
template class mray::hip::TextureHIP<2, Vector2>;
template class mray::hip::TextureHIP<2, Vector3>;
template class mray::hip::TextureHIP<2, Vector4>;

template class mray::hip::TextureHIP<2, uint8_t>;
template class mray::hip::TextureHIP<2, Vector2uc>;
template class mray::hip::TextureHIP<2, Vector3uc>;
template class mray::hip::TextureHIP<2, Vector4uc>;

template class mray::hip::TextureHIP<2, int8_t>;
template class mray::hip::TextureHIP<2, Vector2c>;
template class mray::hip::TextureHIP<2, Vector3c>;
template class mray::hip::TextureHIP<2, Vector4c>;

template class mray::hip::TextureHIP<2, uint16_t>;
template class mray::hip::TextureHIP<2, Vector2us>;
template class mray::hip::TextureHIP<2, Vector3us>;
template class mray::hip::TextureHIP<2, Vector4us>;

template class mray::hip::TextureHIP<2, int16_t>;
template class mray::hip::TextureHIP<2, Vector2s>;
template class mray::hip::TextureHIP<2, Vector3s>;
template class mray::hip::TextureHIP<2, Vector4s>;

template class mray::hip::TextureHIP<2, PixelBC1>;
template class mray::hip::TextureHIP<2, PixelBC2>;
template class mray::hip::TextureHIP<2, PixelBC3>;
template class mray::hip::TextureHIP<2, PixelBC4U>;
template class mray::hip::TextureHIP<2, PixelBC4S>;
template class mray::hip::TextureHIP<2, PixelBC5U>;
template class mray::hip::TextureHIP<2, PixelBC5S>;
template class mray::hip::TextureHIP<2, PixelBC6U>;
template class mray::hip::TextureHIP<2, PixelBC6S>;
template class mray::hip::TextureHIP<2, PixelBC7>;

// Common Textures 3D
template class mray::hip::TextureHIP<3, Float>;
template class mray::hip::TextureHIP<3, Vector2>;
template class mray::hip::TextureHIP<3, Vector3>;
template class mray::hip::TextureHIP<3, Vector4>;

template class mray::hip::TextureHIP<3, uint8_t>;
template class mray::hip::TextureHIP<3, Vector2uc>;
template class mray::hip::TextureHIP<3, Vector3uc>;
template class mray::hip::TextureHIP<3, Vector4uc>;

template class mray::hip::TextureHIP<3, int8_t>;
template class mray::hip::TextureHIP<3, Vector2c>;
template class mray::hip::TextureHIP<3, Vector3c>;
template class mray::hip::TextureHIP<3, Vector4c>;

template class mray::hip::TextureHIP<3, uint16_t>;
template class mray::hip::TextureHIP<3, Vector2us>;
template class mray::hip::TextureHIP<3, Vector3us>;
template class mray::hip::TextureHIP<3, Vector4us>;

template class mray::hip::TextureHIP<3, int16_t>;
template class mray::hip::TextureHIP<3, Vector2s>;
template class mray::hip::TextureHIP<3, Vector3s>;
template class mray::hip::TextureHIP<3, Vector4s>;