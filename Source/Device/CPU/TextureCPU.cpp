#include "TextureCPU.h"
#include "DeviceMemoryCPU.h"
#include "GPUSystemCPU.hpp"

#include "Core/GraphicsFunctions.h"

namespace mray::host
{

template<uint32_t D, class T>
TextureCPU_Normal<D, T>::TextureCPU_Normal(const GPUDeviceCPU& device,
                                           const TextureInitParams<D>& p)
    : gpu(&device)
    , texParams(p)
{
    // Warnings
    if(texParams.normIntegers && !IsNormConvertible)
    {
        MRAY_WARNING_LOG("Requested channel type cannot be converted to normalized form."
                         " Setting \"unormIntegers\" to false");
        texParams.normIntegers = false;
    };

    // Calculate the full size by hand
    static constexpr auto BPP = sizeof(PaddedChannelType);
    uint32_t mipCount = std::min(texParams.mipCount, Graphics::TextureMipCount(texParams.size));
    for(uint32_t i = 0; i < mipCount; i++)
    {
        if constexpr(D == 1)
            size += Graphics::TextureMipSize(texParams.size, i) * BPP;
        else
            size += Graphics::TextureMipSize(texParams.size, i).Multiply() * BPP;
    }
}

template<uint32_t D, class T>
TextureCPU_Normal<D, T>::TextureCPU_Normal(TextureCPU_Normal&& other) noexcept
    : gpu(other.gpu)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{}

template<uint32_t D, class T>
TextureCPU_Normal<D, T>& TextureCPU_Normal<D, T>::operator=(TextureCPU_Normal&& other) noexcept
{
    assert(&other != this);
    gpu = other.gpu;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;
    return *this;
}

template<uint32_t D, class T>
TextureCPU_Normal<D, T>::~TextureCPU_Normal() = default;

template<uint32_t D, class T>
RWTextureRefCPU<D, T> TextureCPU_Normal<D, T>::GenerateRWRef(uint32_t mipLevel)
{
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Requested out of bounds mip level!");

    throw MRayError("Not yet implemented!");
    // TODO: Get the slice with the alignment etc
    // and use it. Shirnk the given span to catch bugs
    // etc..
    //return RWTextureRefCPU<D, T>(Span(dataPtr, size),
    //                             texParams.size);
}

template<uint32_t D, class T>
size_t TextureCPU_Normal<D, T>::Size() const
{
    return size;
}

template<uint32_t D, class T>
size_t TextureCPU_Normal<D, T>::Alignment() const
{
    return alignment;
}

template<uint32_t D, class T>
TextureExtent<D> TextureCPU_Normal<D, T>::Extents() const
{
    return texParams.size;
}

template<uint32_t D, class T>
uint32_t TextureCPU_Normal<D, T>::MipCount() const
{
    return texParams.mipCount;
}

template<uint32_t D, class T>
const GPUDeviceCPU& TextureCPU_Normal<D, T>::Device() const
{
    assert(gpu != nullptr);
    return *gpu;
}

template<uint32_t D, class T>
void TextureCPU_Normal<D, T>::CommitMemory(const GPUQueueCPU&,
                                           const TextureBackingMemoryCPU& deviceMem,
                                           size_t offset)
{
    Span<Byte> data = ToHandleCPU(deviceMem);
    dataPtr = reinterpret_cast<PaddedChannelType*>(data.subspan(offset, size).data());
}

template<uint32_t D, class T>
void TextureCPU_Normal<D, T>::CopyFromAsync(const GPUQueueCPU& queue,
                                            uint32_t mipLevel,
                                            const TextureExtent<D>& offset,
                                            const TextureExtent<D>& fromSize,
                                            Span<const PaddedChannelType> regionFrom)
{
    assert(mipLevel < texParams.mipCount);
    Span<PaddedChannelType> dataSpan = Span<PaddedChannelType>(dataPtr, size);

    size_t mipStartOffset = 0;
    for(uint32_t i = 0; i < mipLevel; i++)
    {
        if constexpr(D == 1)
            mipStartOffset += Graphics::TextureMipSize(texParams.size, i);
        else
            mipStartOffset += Graphics::TextureMipSize(texParams.size, i).Multiply();
    }
    TextureExtent<D> mipSize = Graphics::TextureMipSize(texParams.size, mipLevel);
    assert(offset + fromSize <= mipSize);

    if constexpr(D == 1)
    {
        Span<PaddedChannelType> mipPtr = dataSpan.subspan(mipStartOffset, mipSize);
        mipPtr = mipPtr.subspan(offset, mipSize - offset);
        queue.MemcpyAsync(mipPtr, regionFrom);
    }
    else if constexpr(D == 2)
    {
        size_t mipSizeLinear = mipSize.Multiply();
        size_t toStride = mipSize[0];
        size_t fromStride = fromSize[0];
        Span<PaddedChannelType> mipPtr = dataSpan.subspan(mipStartOffset,
                                                          mipSizeLinear);
        // Y shift
        mipPtr = mipPtr.subspan(offset[1] * mipSize[0]);
        // X shift
        mipPtr = mipPtr.subspan(offset[0]);
        queue.MemcpyAsync2D(mipPtr, toStride, regionFrom, fromStride,
                            fromSize);
    }
    else if constexpr(D == 3)
    {
        // TODO: Change this later to a less-memcpy version
        // Looping over each 2D slice
        size_t mipSizeLinear = mipSize.Multiply();
        Span<PaddedChannelType> mipPtr = dataSpan.subspan(mipStartOffset,
                                                          mipSizeLinear);
        for(uint32_t z = 0; z < fromSize[2]; z++)
        {
            Vector2ui sliceSize = Vector2ui(mipSize);
            Vector2ui fromSliceSize = Vector2ui(fromSize);

            size_t toStride = sliceSize[0];
            size_t fromStride = fromSize[0];
            Span<PaddedChannelType> toSlice = mipPtr.subspan(z * sliceSize.Multiply(),
                                                             sliceSize.Multiply());
            Span<const PaddedChannelType> fromSlice = regionFrom.subspan(z * fromSliceSize.Multiply(),
                                                                         fromSliceSize.Multiply());
            // Y shift
            toSlice = toSlice.subspan(offset[1] * mipSize[0]);
            // X shift
            toSlice = toSlice.subspan(offset[0]);
            queue.MemcpyAsync2D(toSlice, toStride, fromSlice, fromStride,
                                Vector2ui(fromSize));
        }
    }
}

template<uint32_t D, class T>
void TextureCPU_Normal<D, T>::CopyToAsync(Span<PaddedChannelType> regionTo,
                                          const GPUQueueCPU& queue,
                                          uint32_t mipLevel,
                                          const TextureExtent<D>& offset,
                                          const TextureExtent<D>& toSize) const
{
    assert(mipLevel < texParams.mipCount);
    Span<PaddedChannelType> dataSpan = Span<PaddedChannelType>(dataPtr, size);

    size_t mipStartOffset = 0;
    for(uint32_t i = 0; i < mipLevel; i++)
    {
        if constexpr(D == 1)
            mipStartOffset += Graphics::TextureMipSize(texParams.size, i);
        else
            mipStartOffset += Graphics::TextureMipSize(texParams.size, i).Multiply();
    }
    TextureExtent<D> mipSize = Graphics::TextureMipSize(texParams.size, mipLevel);
    assert(offset + toSize < mipSize);

    if constexpr(D == 1)
    {
        Span<PaddedChannelType> mipPtr = dataSpan.subspan(mipStartOffset, mipSize);
        mipPtr = mipPtr.subspan(offset, mipSize - offset);
        queue.MemcpyAsync(regionTo, ToConstSpan(mipPtr));
    }
    else if constexpr(D == 2)
    {
        size_t mipSizeLinear = mipSize.Multiply();
        size_t fromStride = mipSize[0];
        size_t toStride = toSize[0];
        Span<PaddedChannelType> mipPtr = dataSpan.subspan(mipStartOffset,
                                                          mipSizeLinear);
        // Y shift
        mipPtr = mipPtr.subspan(offset[1] * mipSize[0]);
        // X shift
        mipPtr = mipPtr.subspan(offset[0]);
        queue.MemcpyAsync2D(regionTo, toStride, ToConstSpan(mipPtr), fromStride, toSize);
    }
    else if constexpr(D == 3)
    {
        // TODO: Change this later to a less-memcpy version
        // Looping over each 2D slice
        size_t mipSizeLinear = mipSize.Multiply();
        Span<PaddedChannelType> mipPtr = dataSpan.subspan(mipStartOffset,
                                                          mipSizeLinear);
        for(uint32_t z = offset[2]; z < offset[2] + toSize[2]; z++)
        {
            Vector2ui sliceSize = Vector2ui(mipSize);
            Vector2ui toSliceSize = Vector2ui(toSize);

            size_t fromStride = sliceSize[0];
            size_t toStride = toSize[0];
            Span<PaddedChannelType> fromSlice = mipPtr.subspan(z * sliceSize.Multiply(),
                                                               sliceSize.Multiply());
            Span<PaddedChannelType> toSlice = regionTo.subspan(z * toSliceSize.Multiply(),
                                                               toSliceSize.Multiply());
            // Y shift
            fromSlice = fromSlice.subspan(offset[1] * mipSize[0]);
            // X shift
            fromSlice = fromSlice.subspan(offset[0]);
            queue.MemcpyAsync2D(toSlice, toStride, ToConstSpan(fromSlice), fromStride,
                                Vector2ui(toSize));
        }
    }
}

template<class T>
TextureCPU_BC<T>::TextureCPU_BC(const GPUDeviceCPU&,
                                const TextureInitParams<2>&)
{
    // Unreachable code warning MSVC?
    //throw MRayError("CPU Device does not support BC textures!");
}

template<class T>
size_t TextureCPU_BC<T>::Size() const
{
    return size;
}

template<class T>
size_t TextureCPU_BC<T>::Alignment() const

{
    return alignment;
}

template<class T>
TextureExtent<2> TextureCPU_BC<T>::Extents() const
{
    return texParams.size;
}

template<class T>
uint32_t TextureCPU_BC<T>::MipCount() const
{
    return texParams.mipCount;
}

template<class T>
const GPUDeviceCPU& TextureCPU_BC<T>::Device() const
{
    assert(gpu != nullptr);
    return *gpu;
}

template<class T>
void TextureCPU_BC<T>::CommitMemory(const GPUQueueCPU&,
                                    const TextureBackingMemoryCPU&,
                                    size_t)
{
    throw MRayError("CPU Device does not support BC textures!");
}

template<class T>
void TextureCPU_BC<T>::CopyFromAsync(const GPUQueueCPU&, uint32_t,
                                     const TextureExtent<2>&, const TextureExtent<2>&,
                                     Span<const PaddedChannelType>)
{
    throw MRayError("CPU Device does not support BC textures!");
}

template<class T>
void TextureCPU_BC<T>::CopyToAsync(Span<PaddedChannelType>, const GPUQueueCPU&,
                                    uint32_t, const TextureExtent<2>&,
                                    const TextureExtent<2>&) const
{
    throw MRayError("CPU Device does not support BC textures!");
}

TextureBackingMemoryCPU::TextureBackingMemoryCPU(const GPUDeviceCPU& gpu)
    : gpu(&gpu)
    , size(0)
    , allocSize(0)
    , memPtr(nullptr)
{}

TextureBackingMemoryCPU::TextureBackingMemoryCPU(const GPUDeviceCPU& device, size_t sizeInBytes)
    : TextureBackingMemoryCPU(device)
{
    size = sizeInBytes;
    allocSize = Math::NextMultiple(size, MemAlloc::DefaultSystemAlignment());
    memPtr = AlignedAllocate(allocSize, MemAlloc::DefaultSystemAlignment());
}

TextureBackingMemoryCPU::TextureBackingMemoryCPU(TextureBackingMemoryCPU&& other) noexcept
    : gpu(other.gpu)
    , size(other.size)
    , allocSize(other.allocSize)
    , memPtr(other.memPtr)
{
    other.size = 0;
    other.allocSize = 0;
    other.memPtr = nullptr;
}

TextureBackingMemoryCPU::~TextureBackingMemoryCPU()
{
    AlignedFree(memPtr, allocSize, MemAlloc::DefaultSystemAlignment());
}

TextureBackingMemoryCPU& TextureBackingMemoryCPU::operator=(TextureBackingMemoryCPU&& other) noexcept
{
    if(memPtr) AlignedFree(memPtr, allocSize, MemAlloc::DefaultSystemAlignment());

    gpu = other.gpu;
    size = other.size;
    allocSize = other.allocSize;
    memPtr = other.memPtr;
    //
    other.size = 0;
    other.allocSize = 0;
    other.memPtr = nullptr;
    return *this;
}

void TextureBackingMemoryCPU::ResizeBuffer(size_t newSize)
{
    TextureBackingMemoryCPU newMem(*gpu, newSize);
    std::memcpy(newMem.memPtr, memPtr, std::min(newSize, size));
    *this = std::move(newMem);
}

const GPUDeviceCPU& TextureBackingMemoryCPU::Device() const
{
    return *gpu;
}

size_t TextureBackingMemoryCPU::Size() const
{
    return size;
}

}

// Common Textures 1D
template class mray::host::TextureCPU_Normal<1, Float>;
template class mray::host::TextureCPU_Normal<1, Vector2>;
template class mray::host::TextureCPU_Normal<1, Vector3>;
template class mray::host::TextureCPU_Normal<1, Vector4>;

template class mray::host::TextureCPU_Normal<1, uint8_t>;
template class mray::host::TextureCPU_Normal<1, Vector2uc>;
template class mray::host::TextureCPU_Normal<1, Vector3uc>;
template class mray::host::TextureCPU_Normal<1, Vector4uc>;

template class mray::host::TextureCPU_Normal<1, int8_t>;
template class mray::host::TextureCPU_Normal<1, Vector2c>;
template class mray::host::TextureCPU_Normal<1, Vector3c>;
template class mray::host::TextureCPU_Normal<1, Vector4c>;

template class mray::host::TextureCPU_Normal<1, uint16_t>;
template class mray::host::TextureCPU_Normal<1, Vector2us>;
template class mray::host::TextureCPU_Normal<1, Vector3us>;
template class mray::host::TextureCPU_Normal<1, Vector4us>;

template class mray::host::TextureCPU_Normal<1, int16_t>;
template class mray::host::TextureCPU_Normal<1, Vector2s>;
template class mray::host::TextureCPU_Normal<1, Vector3s>;
template class mray::host::TextureCPU_Normal<1, Vector4s>;

// Common Textures 2D
template class mray::host::TextureCPU_Normal<2, Float>;
template class mray::host::TextureCPU_Normal<2, Vector2>;
template class mray::host::TextureCPU_Normal<2, Vector3>;
template class mray::host::TextureCPU_Normal<2, Vector4>;

template class mray::host::TextureCPU_Normal<2, uint8_t>;
template class mray::host::TextureCPU_Normal<2, Vector2uc>;
template class mray::host::TextureCPU_Normal<2, Vector3uc>;
template class mray::host::TextureCPU_Normal<2, Vector4uc>;

template class mray::host::TextureCPU_Normal<2, int8_t>;
template class mray::host::TextureCPU_Normal<2, Vector2c>;
template class mray::host::TextureCPU_Normal<2, Vector3c>;
template class mray::host::TextureCPU_Normal<2, Vector4c>;

template class mray::host::TextureCPU_Normal<2, uint16_t>;
template class mray::host::TextureCPU_Normal<2, Vector2us>;
template class mray::host::TextureCPU_Normal<2, Vector3us>;
template class mray::host::TextureCPU_Normal<2, Vector4us>;

template class mray::host::TextureCPU_Normal<2, int16_t>;
template class mray::host::TextureCPU_Normal<2, Vector2s>;
template class mray::host::TextureCPU_Normal<2, Vector3s>;
template class mray::host::TextureCPU_Normal<2, Vector4s>;

template class mray::host::TextureCPU_BC<PixelBC1>;
template class mray::host::TextureCPU_BC<PixelBC2>;
template class mray::host::TextureCPU_BC<PixelBC3>;
template class mray::host::TextureCPU_BC<PixelBC4U>;
template class mray::host::TextureCPU_BC<PixelBC4S>;
template class mray::host::TextureCPU_BC<PixelBC5U>;
template class mray::host::TextureCPU_BC<PixelBC5S>;
template class mray::host::TextureCPU_BC<PixelBC6U>;
template class mray::host::TextureCPU_BC<PixelBC6S>;
template class mray::host::TextureCPU_BC<PixelBC7>;

// Common Textures 3D
template class mray::host::TextureCPU_Normal<3, Float>;
template class mray::host::TextureCPU_Normal<3, Vector2>;
template class mray::host::TextureCPU_Normal<3, Vector3>;
template class mray::host::TextureCPU_Normal<3, Vector4>;

template class mray::host::TextureCPU_Normal<3, uint8_t>;
template class mray::host::TextureCPU_Normal<3, Vector2uc>;
template class mray::host::TextureCPU_Normal<3, Vector3uc>;
template class mray::host::TextureCPU_Normal<3, Vector4uc>;

template class mray::host::TextureCPU_Normal<3, int8_t>;
template class mray::host::TextureCPU_Normal<3, Vector2c>;
template class mray::host::TextureCPU_Normal<3, Vector3c>;
template class mray::host::TextureCPU_Normal<3, Vector4c>;

template class mray::host::TextureCPU_Normal<3, uint16_t>;
template class mray::host::TextureCPU_Normal<3, Vector2us>;
template class mray::host::TextureCPU_Normal<3, Vector3us>;
template class mray::host::TextureCPU_Normal<3, Vector4us>;

template class mray::host::TextureCPU_Normal<3, int16_t>;
template class mray::host::TextureCPU_Normal<3, Vector2s>;
template class mray::host::TextureCPU_Normal<3, Vector3s>;
template class mray::host::TextureCPU_Normal<3, Vector4s>;

// Common Textures 1D
template class mray::host::TextureCPU<1, Float>;
template class mray::host::TextureCPU<1, Vector2>;
template class mray::host::TextureCPU<1, Vector3>;
template class mray::host::TextureCPU<1, Vector4>;

template class mray::host::TextureCPU<1, uint8_t>;
template class mray::host::TextureCPU<1, Vector2uc>;
template class mray::host::TextureCPU<1, Vector3uc>;
template class mray::host::TextureCPU<1, Vector4uc>;

template class mray::host::TextureCPU<1, int8_t>;
template class mray::host::TextureCPU<1, Vector2c>;
template class mray::host::TextureCPU<1, Vector3c>;
template class mray::host::TextureCPU<1, Vector4c>;

template class mray::host::TextureCPU<1, uint16_t>;
template class mray::host::TextureCPU<1, Vector2us>;
template class mray::host::TextureCPU<1, Vector3us>;
template class mray::host::TextureCPU<1, Vector4us>;

template class mray::host::TextureCPU<1, int16_t>;
template class mray::host::TextureCPU<1, Vector2s>;
template class mray::host::TextureCPU<1, Vector3s>;
template class mray::host::TextureCPU<1, Vector4s>;

// Common Textures 2D
template class mray::host::TextureCPU<2, Float>;
template class mray::host::TextureCPU<2, Vector2>;
template class mray::host::TextureCPU<2, Vector3>;
template class mray::host::TextureCPU<2, Vector4>;

template class mray::host::TextureCPU<2, uint8_t>;
template class mray::host::TextureCPU<2, Vector2uc>;
template class mray::host::TextureCPU<2, Vector3uc>;
template class mray::host::TextureCPU<2, Vector4uc>;

template class mray::host::TextureCPU<2, int8_t>;
template class mray::host::TextureCPU<2, Vector2c>;
template class mray::host::TextureCPU<2, Vector3c>;
template class mray::host::TextureCPU<2, Vector4c>;

template class mray::host::TextureCPU<2, uint16_t>;
template class mray::host::TextureCPU<2, Vector2us>;
template class mray::host::TextureCPU<2, Vector3us>;
template class mray::host::TextureCPU<2, Vector4us>;

template class mray::host::TextureCPU<2, int16_t>;
template class mray::host::TextureCPU<2, Vector2s>;
template class mray::host::TextureCPU<2, Vector3s>;
template class mray::host::TextureCPU<2, Vector4s>;

template class mray::host::TextureCPU<2, PixelBC1>;
template class mray::host::TextureCPU<2, PixelBC2>;
template class mray::host::TextureCPU<2, PixelBC3>;
template class mray::host::TextureCPU<2, PixelBC4U>;
template class mray::host::TextureCPU<2, PixelBC4S>;
template class mray::host::TextureCPU<2, PixelBC5U>;
template class mray::host::TextureCPU<2, PixelBC5S>;
template class mray::host::TextureCPU<2, PixelBC6U>;
template class mray::host::TextureCPU<2, PixelBC6S>;
template class mray::host::TextureCPU<2, PixelBC7>;

// Common Textures 3D
template class mray::host::TextureCPU<3, Float>;
template class mray::host::TextureCPU<3, Vector2>;
template class mray::host::TextureCPU<3, Vector3>;
template class mray::host::TextureCPU<3, Vector4>;

template class mray::host::TextureCPU<3, uint8_t>;
template class mray::host::TextureCPU<3, Vector2uc>;
template class mray::host::TextureCPU<3, Vector3uc>;
template class mray::host::TextureCPU<3, Vector4uc>;

template class mray::host::TextureCPU<3, int8_t>;
template class mray::host::TextureCPU<3, Vector2c>;
template class mray::host::TextureCPU<3, Vector3c>;
template class mray::host::TextureCPU<3, Vector4c>;

template class mray::host::TextureCPU<3, uint16_t>;
template class mray::host::TextureCPU<3, Vector2us>;
template class mray::host::TextureCPU<3, Vector3us>;
template class mray::host::TextureCPU<3, Vector4us>;

template class mray::host::TextureCPU<3, int16_t>;
template class mray::host::TextureCPU<3, Vector2s>;
template class mray::host::TextureCPU<3, Vector3s>;
template class mray::host::TextureCPU<3, Vector4s>;