#include "TextureCPU.h"
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
}

template<uint32_t D, class T>
TextureCPU_Normal<D, T>::TextureCPU_Normal(TextureCPU_Normal&& other) noexcept
    : gpu(other.gpu)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{
}

template<uint32_t D, class T>
TextureCPU_Normal<D, T>& TextureCPU_Normal<D, T>::operator=(TextureCPU_Normal&& other) noexcept
{

    gpu = other.gpu;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;
    return *this;
}

template<uint32_t D, class T>
TextureCPU_Normal<D, T>::~TextureCPU_Normal()
{
}

template<uint32_t D, class T>
RWTextureRefCPU<D, T> TextureCPU_Normal<D, T>::GenerateRWRef(uint32_t mipLevel)
{
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Requested out of bounds mip level!");

    return RWTextureRefCPU<D, T>();
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
void TextureCPU_Normal<D, T>::CommitMemory(const GPUQueueCPU& queue,
                                           const TextureBackingMemoryCPU& deviceMem,
                                           size_t offset)
{
}

template<uint32_t D, class T>
void TextureCPU_Normal<D, T>::CopyFromAsync(const GPUQueueCPU& queue,
                                            uint32_t mipLevel,
                                            const TextureExtent<D>& offset,
                                            const TextureExtent<D>& sizes,
                                            Span<const PaddedChannelType> regionFrom)
{
}

template<class T>
TextureCPU_BC<T>::TextureCPU_BC(const GPUDeviceCPU& device,
                                  const TextureInitParams<2>& p)
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
}

template<class T>
TextureCPU_BC<T>::TextureCPU_BC(TextureCPU_BC&& other) noexcept
    : gpu(other.gpu)
    , texParams(other.texParams)
    , size(other.size)
    , alignment(other.alignment)
    , allocated(other.allocated)
{}

template<class T>
TextureCPU_BC<T>& TextureCPU_BC<T>::operator=(TextureCPU_BC&& other) noexcept
{
    gpu = other.gpu;
    texParams = other.texParams;
    allocated = other.allocated;
    size = other.size;
    alignment = other.alignment;
    return *this;
}

template<class T>
TextureCPU_BC<T>::~TextureCPU_BC()
{}

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
void TextureCPU_BC<T>::CommitMemory(const GPUQueueCPU& queue,
                                    const TextureBackingMemoryCPU& deviceMem,
                                    size_t offset)
{
}

template<class T>
void TextureCPU_BC<T>::CopyFromAsync(const GPUQueueCPU& queue,
                                     uint32_t mipLevel,
                                     const TextureExtent<2>& offset,
                                     const TextureExtent<2>& sizes,
                                     Span<const PaddedChannelType> regionFrom)
{
    // Silently ignore the mip if out of bounds
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Out of bounds mip level is requested!");
}

template<class T>
void TextureCPU_BC<T>::CopyToAsync(Span<PaddedChannelType> regionFrom,
                                   const GPUQueueCPU& queue,
                                   uint32_t mipLevel,
                                   const TextureExtent<2>& offset,
                                   const TextureExtent<2>& sizes) const
{
    // Silently ignore the mip if out of bounds
    if(mipLevel >= texParams.mipCount)
        throw MRayError("Out of bounds mip level is requested!");
}

TextureBackingMemoryCPU::TextureBackingMemoryCPU(const GPUDeviceCPU& gpu)
    : gpu(&gpu)
    , size(0)
    , allocSize(0)
{}

TextureBackingMemoryCPU::TextureBackingMemoryCPU(const GPUDeviceCPU& device, size_t sizeInBytes)
    : TextureBackingMemoryCPU(device)
{}

TextureBackingMemoryCPU::TextureBackingMemoryCPU(TextureBackingMemoryCPU&& other) noexcept
    : gpu(other.gpu)
    , size(other.size)
    , allocSize(other.allocSize)
{
    other.size = 0;
    other.allocSize = 0;
}

TextureBackingMemoryCPU::~TextureBackingMemoryCPU()
{
}

TextureBackingMemoryCPU& TextureBackingMemoryCPU::operator=(TextureBackingMemoryCPU&& other) noexcept
{
    return *this;
}

void TextureBackingMemoryCPU::ResizeBuffer(size_t newSize)
{
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