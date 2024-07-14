
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


template <uint32_t D, class T>
RWTextureRefCUDA<D, T>::RWTextureRefCUDA(cudaSurfaceObject_t sIn)
    : s(sIn)
{}

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>::RWTextureRefCUDA(RWTextureRefCUDA&& other)
    : s(std::exchange(other.s, cudaSurfaceObject_t(0)))
{}

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>& RWTextureRefCUDA<D, T>::operator=(RWTextureRefCUDA&& other)
{
    assert(this != &other);
    CUDA_CHECK(cudaDestroySurfaceObject(s));
    s = std::exchange(other.s, cudaSurfaceObject_t(0));
    return *this;
}

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>::~RWTextureRefCUDA()
{
    CUDA_CHECK(cudaDestroySurfaceObject(s));
}

template <uint32_t D, class T>
RWTextureViewCUDA<D, T> RWTextureRefCUDA<D, T>::View() const
{
    return RWTextureViewCUDA<D, T>(s);
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
    , allocated(other.allocated)
    , size(other.size)
    , alignment(other.alignment)

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
template<class QT>
requires(std::is_same_v<QT, T>)
TextureViewCUDA<D, QT> TextureCUDA_Normal<D, T>::View() const
{
    // Normalize integers requested bu view is created with the same type
    if(texParams.normIntegers)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for normalized integers");
    return TextureViewCUDA<D, QT>(tex);
}

template<uint32_t D, class T>
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (VectorTypeToChannels::Find<T> ==
          VectorTypeToChannels::Find<QT>))
TextureViewCUDA<D, QT> TextureCUDA_Normal<D, T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                 std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(texParams.normCoordinates && !IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for normalized integers");
    else if(texParams.normCoordinates && IsFloatType)
        return TextureViewCUDA<D, QT>(tex);
    else
    {
        // Now we have integer types and these will not be fetched
        // as normalized floats. This means type conversion either narrowing
        // or expanding. (i.e., "short -> int" or "int -> short").
        //
        // Do not support these currently, I dunno if it compiles down to
        // auto narrowing or texture fetch functions will fail.
        //
        // TODO: change this later if all is OK
        throw MRayError("Unable to create a view of texture. "
                        "Any type conversion (narrowing or expanding) is not supported on textures."
                        " This function should only be called for normalized integers with \"Float\" types");
    }
}

template<uint32_t D, class T>
RWTextureRefCUDA<D, T> TextureCUDA_Normal<D, T>::GenerateRWRef(uint32_t mipLevel)
{
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Requested out of bounds mip level!");

    // TODO: Check if we are owning this cudaArray_t.
    // Since it is a "get" function we do not own this I guess
    cudaArray_t mipLevelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&mipLevelArray, data,
                                          static_cast<int>(mipLevel)));

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
    cudaArray_t levelArray;
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

    using namespace MathFunctions;
    Vector2ui paddedSize = Vector2ui(NextMultiple(p.size[0], BC_BLOCK_SIZE),
                                     NextMultiple(p.size[1], BC_BLOCK_SIZE));
    if(paddedSize != p.size)
    {
        MRAY_WARNING_LOG("BC texture size is not multiple of {}! "
                         "Texture may be skewed!", BC_BLOCK_SIZE);
    }
    cudaExtent extent = MakeCudaExtent<2, 0u>(paddedSize);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<CudaTypeEnum>();
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_MEM_THROW(cudaMallocMipmappedArray(&data, &channelDesc, extent, p.mipCount,
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
    // Always normalized float for block compressed texture
    tDesc.readMode = cudaReadModeNormalizedFloat;
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
    , allocated(other.allocated)
    , size(other.size)
    , alignment(other.alignment)

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
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (BCTypeToChannels<T>() == VectorTypeToChannels::Find<QT>))
TextureViewCUDA<2, QT> TextureCUDA_BC<T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                  std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(!IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for block compressed textures (BC1-7)");
    return TextureViewCUDA<2, QT>(tex);
};

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
    cudaArray_t levelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyDefault;
    p.extent = MakeCudaExtent<2, 1>(sizes);

    p.dstArray = levelArray;
    p.dstPos = MakeCudaPos<2, 0>(offset);

    p.srcPos = make_cudaPos(0, 0, 0);
    //
    void* ptr = const_cast<Byte*>(reinterpret_cast<const Byte*>(regionFrom.data()));
    p.srcPtr = make_cudaPitchedPtr(ptr,
                                   p.extent.width * sizeof(PaddedChannelType),
                                   p.extent.width, p.extent.height);
    CUDA_CHECK(cudaMemcpy3DAsync(&p, ToHandleCUDA(queue)));
}

}
