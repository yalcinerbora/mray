#pragma once

#include "DeviceMemoryCPU.h"
#include "TextureViewCPU.h"
#include "DefinitionsCPU.h"
#include "GPUSystemCPU.h"

#include "../GPUTypes.h"

#include "Core/Definitions.h"

namespace mray::host
{

using PixelTypeToEnum = TypeFinder::T_VMapper:: template Map
<
    TypeFinder::T_VMapper::template TVPair<uint8_t, MRayPixelEnum::MR_R8_UNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector2uc, MRayPixelEnum::MR_RG8_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<Vector4uc, MRayPixelEnum::MR_RGB8_UNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector4uc, MRayPixelEnum::MR_RGBA8_UNORM>,

    TypeFinder::T_VMapper::template TVPair<uint16_t, MRayPixelEnum::MR_R16_UNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector2us, MRayPixelEnum::MR_RG16_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<Vector4us, MRayPixelEnum::MR_RGB16_UNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector4us, MRayPixelEnum::MR_RGBA16_UNORM>,

    TypeFinder::T_VMapper::template TVPair<int8_t, MRayPixelEnum::MR_R8_SNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector2c, MRayPixelEnum::MR_RG8_SNORM>,
    //TypeFinder::T_VMapper::template TVPair<Vector4c, MRayPixelEnum::MR_RGB8_SNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector4c, MRayPixelEnum::MR_RGBA8_SNORM>,

    TypeFinder::T_VMapper::template TVPair<int16_t, MRayPixelEnum::MR_R16_SNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector2s, MRayPixelEnum::MR_RG16_SNORM>,
    //TypeFinder::T_VMapper::template TVPair<Vector4s, MRayPixelEnum::MR_RGB16_SNORM>,
    TypeFinder::T_VMapper::template TVPair<Vector4s, MRayPixelEnum::MR_RGBA16_SNORM>,

    //TypeFinder::T_VMapper::template TVPair<half, MRayPixelEnum::MR_R_HALF>,
    //TypeFinder::T_VMapper::template TVPair<Vector2uh, MRayPixelEnum::MR_RG_HALF>,
    //TypeFinder::T_VMapper::template TVPair<Vector4uh, MRayPixelEnum::MR_RGB_HALF>,
    //TypeFinder::T_VMapper::template TVPair<Vector4uh, MRayPixelEnum::MR_RGBA_HALF>,

    TypeFinder::T_VMapper::template TVPair<Float, MRayPixelEnum::MR_R_FLOAT>,
    TypeFinder::T_VMapper::template TVPair<Vector2f, MRayPixelEnum::MR_RG_FLOAT>,
    //TypeFinder::T_VMapper::template TVPair<Vector3f, MRayPixelEnum::MR_RGB_FLOAT>,
    TypeFinder::T_VMapper::template TVPair<Vector4f, MRayPixelEnum::MR_RGBA_FLOAT>

    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC1_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC2_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC3_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC4_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC4_SNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC5_UNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC5_SNORM>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC6H_UFLOAT>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC6H_SFLOAT>,
    //TypeFinder::T_VMapper::template TVPair<..., MRayPixelEnum::MR_BC7_UNORM>

>;

class TextureBackingMemoryCPU;

template<uint32_t D, class T>
class TextureCPU;

template <class T>
constexpr bool IsNormConvertibleCPU()
{
    // 32-bit types are not norm convertible,
    // so removed these from this function
    //
    // YOLO
    return (std::is_same_v<T, uint16_t>     ||
            std::is_same_v<T, Vector2us>    ||
            std::is_same_v<T, Vector3us>    ||
            std::is_same_v<T, Vector4us>    ||

            std::is_same_v<T, int16_t>      ||
            std::is_same_v<T, Vector2s>     ||
            std::is_same_v<T, Vector3s>     ||
            std::is_same_v<T, Vector4s>     ||

            std::is_same_v<T, uint8_t>      ||
            std::is_same_v<T, Vector2uc>    ||
            std::is_same_v<T, Vector3uc>    ||
            std::is_same_v<T, Vector4uc>    ||

            std::is_same_v<T, int8_t>       ||
            std::is_same_v<T, Vector2c>     ||
            std::is_same_v<T, Vector3c>     ||
            std::is_same_v<T, Vector4c>);
}

template <class T>
constexpr uint32_t BCTypeToBlockSize()
{
    // https://developer.nvidia.com/blog/revealing-new-features-in-the-cuda-11-5-toolkit/
    if constexpr(std::is_same_v<T, PixelBC1>  ||
                 std::is_same_v<T, PixelBC4U> ||
                 std::is_same_v<T, PixelBC4S>)
    {
        return 8;
    }
    else if constexpr(std::is_same_v<T, PixelBC2>  ||
                      std::is_same_v<T, PixelBC3>  ||
                      std::is_same_v<T, PixelBC5U> ||
                      std::is_same_v<T, PixelBC5S> ||
                      std::is_same_v<T, PixelBC6U> ||
                      std::is_same_v<T, PixelBC6S> ||
                      std::is_same_v<T, PixelBC7>)
    {
        return 16;
    }
    else static_assert(std::is_same_v<T, PixelBC1>,
                       "Unknown Block Compressed Format!");
}

// This is unnecesary layer but GPUs need it so
// we just delegate the data to the actual non-owning view
template<uint32_t D, class T>
class RWTextureRefCPU
{
    friend class TextureCPU<D, T>;
    static constexpr uint32_t C = PixelTypeToChannels<T>();
    using PaddedChannelType = PaddedChannel<C, T>;

    private:
    Span<PaddedChannelType> data;
    TextureExtent<D>        dim;

    // Hide this, only texture class can create
                            RWTextureRefCPU(Span<PaddedChannelType> data,
                                            TextureExtent<D> dim);
    public:
    // Constructors & Destructor
                            RWTextureRefCPU(const RWTextureRefCPU&) = delete;
                            RWTextureRefCPU(RWTextureRefCPU&&) = default;
    RWTextureRefCPU&        operator=(const RWTextureRefCPU&) = delete;
    RWTextureRefCPU&        operator=(RWTextureRefCPU&&) = default;
                            ~RWTextureRefCPU() = default;
    //
    RWTextureViewCPU<D, T>  View() const;
};

template<uint32_t D, class T>
class TextureCPU
{
    public:
    static constexpr uint32_t ChannelCount  = PixelTypeToChannels<T>();
    static constexpr bool IsNormConvertible = IsNormConvertibleCPU<T>();
    static constexpr uint32_t Dims          = D;
    static constexpr bool IsBlockCompressed = IsBlockCompressedPixel<T>;

    using Type              = T;
    using PaddedChannelType = PaddedChannel<ChannelCount, T>;

    // Sanity Check
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");

    private:
    const GPUDeviceCPU*     gpu;
    // TODO: Make this tex to match generic buffer size;
    PaddedChannelType*      dataPtr     = nullptr;
    TextureInitParams<D>    texParams;
    uint64_t                padding     = 0;

    // Allocation related
    size_t  size        = 0;
    size_t  alignment   = 64_KiB;
    bool    allocated   = false;


    protected:
    public:
    // Constructors & Destructor
                TextureCPU() = delete;
                TextureCPU(const GPUDeviceCPU& device,
                           const TextureInitParams<D>& p);
                TextureCPU(const TextureCPU&) = delete;
                TextureCPU(TextureCPU&&) noexcept;
    TextureCPU& operator=(const TextureCPU&) = delete;
    TextureCPU& operator=(TextureCPU&&) noexcept;
                ~TextureCPU();

    // Direct view conversion (simple case)
    template<class QT>
    requires(!IsBlockCompressedPixel<T> && std::is_same_v<QT, T>)
    TextureViewCPU<D, QT>  View() const;

    template<class QT>
    requires(!IsBlockCompressedPixel<T> && !std::is_same_v<QT, T> &&
             (PixelTypeToChannels<T>() == PixelTypeToChannels<QT>()))
    TextureViewCPU<D, QT>   View() const;

    template<class QT>
    requires(IsBlockCompressedPixel<T>)
    TextureViewCPU<D, QT>   View() const;

    RWTextureRefCPU<D, T>   GenerateRWRef(uint32_t mipLevel);

    size_t                  Size() const;
    size_t                  Alignment() const;

    TextureExtent<D>        Extents() const;
    uint32_t                MipCount() const;
    const GPUDeviceCPU&     Device() const;

    void                    CommitMemory(const GPUQueueCPU& queue,
                                         const TextureBackingMemoryCPU& deviceMem,
                                         size_t offset);
    void                    CopyFromAsync(const GPUQueueCPU& queue,
                                          uint32_t mipLevel,
                                          const TextureExtent<D>& offset,
                                          const TextureExtent<D>& size,
                                          Span<const PaddedChannelType> regionFrom);
    void                    CopyToAsync(Span<PaddedChannelType> regionTo,
                                        const GPUQueueCPU& queue,
                                        uint32_t mipLevel,
                                        const TextureExtent<D>& offset,
                                        const TextureExtent<D>& toSize) const;
};

class TextureBackingMemoryCPU
{
    friend Span<Byte> ToHandleCPU(const TextureBackingMemoryCPU&);

    private:
    const GPUDeviceCPU* gpu;
    size_t              size;
    size_t              allocSize;
    void*               memPtr;

    public:
    // Constructors & Destructor
                                TextureBackingMemoryCPU(const GPUDeviceCPU& device);
                                TextureBackingMemoryCPU(const GPUDeviceCPU& device, size_t sizeInBytes);
                                TextureBackingMemoryCPU(const TextureBackingMemoryCPU&) = delete;
                                TextureBackingMemoryCPU(TextureBackingMemoryCPU&&) noexcept;
                                ~TextureBackingMemoryCPU();
    TextureBackingMemoryCPU&    operator=(const TextureBackingMemoryCPU&) = delete;
    TextureBackingMemoryCPU&    operator=(TextureBackingMemoryCPU&&) noexcept;

    void                        ResizeBuffer(size_t newSize);
    const GPUDeviceCPU&         Device() const;
    size_t                      Size() const;
};

inline
Span<Byte> ToHandleCPU(const TextureBackingMemoryCPU& mem)
{
    return Span<Byte>(static_cast<Byte*>(mem.memPtr), mem.size);
}

}

#include "TextureCPU.hpp"

// Common Textures 1D
extern template class mray::host::TextureCPU<1, Float>;
extern template class mray::host::TextureCPU<1, Vector2>;
extern template class mray::host::TextureCPU<1, Vector3>;
extern template class mray::host::TextureCPU<1, Vector4>;

extern template class mray::host::TextureCPU<1, uint8_t>;
extern template class mray::host::TextureCPU<1, Vector2uc>;
extern template class mray::host::TextureCPU<1, Vector3uc>;
extern template class mray::host::TextureCPU<1, Vector4uc>;

extern template class mray::host::TextureCPU<1, int8_t>;
extern template class mray::host::TextureCPU<1, Vector2c>;
extern template class mray::host::TextureCPU<1, Vector3c>;
extern template class mray::host::TextureCPU<1, Vector4c>;

extern template class mray::host::TextureCPU<1, uint16_t>;
extern template class mray::host::TextureCPU<1, Vector2us>;
extern template class mray::host::TextureCPU<1, Vector3us>;
extern template class mray::host::TextureCPU<1, Vector4us>;

extern template class mray::host::TextureCPU<1, int16_t>;
extern template class mray::host::TextureCPU<1, Vector2s>;
extern template class mray::host::TextureCPU<1, Vector3s>;
extern template class mray::host::TextureCPU<1, Vector4s>;

// Common Textures 2D
extern template class mray::host::TextureCPU<2, Float>;
extern template class mray::host::TextureCPU<2, Vector2>;
extern template class mray::host::TextureCPU<2, Vector3>;
extern template class mray::host::TextureCPU<2, Vector4>;

extern template class mray::host::TextureCPU<2, uint8_t>;
extern template class mray::host::TextureCPU<2, Vector2uc>;
extern template class mray::host::TextureCPU<2, Vector3uc>;
extern template class mray::host::TextureCPU<2, Vector4uc>;

extern template class mray::host::TextureCPU<2, int8_t>;
extern template class mray::host::TextureCPU<2, Vector2c>;
extern template class mray::host::TextureCPU<2, Vector3c>;
extern template class mray::host::TextureCPU<2, Vector4c>;

extern template class mray::host::TextureCPU<2, uint16_t>;
extern template class mray::host::TextureCPU<2, Vector2us>;
extern template class mray::host::TextureCPU<2, Vector3us>;
extern template class mray::host::TextureCPU<2, Vector4us>;

extern template class mray::host::TextureCPU<2, int16_t>;
extern template class mray::host::TextureCPU<2, Vector2s>;
extern template class mray::host::TextureCPU<2, Vector3s>;
extern template class mray::host::TextureCPU<2, Vector4s>;

// Common Textures 3D
extern template class mray::host::TextureCPU<3, Float>;
extern template class mray::host::TextureCPU<3, Vector2>;
extern template class mray::host::TextureCPU<3, Vector3>;
extern template class mray::host::TextureCPU<3, Vector4>;

extern template class mray::host::TextureCPU<3, uint8_t>;
extern template class mray::host::TextureCPU<3, Vector2uc>;
extern template class mray::host::TextureCPU<3, Vector3uc>;
extern template class mray::host::TextureCPU<3, Vector4uc>;

extern template class mray::host::TextureCPU<3, int8_t>;
extern template class mray::host::TextureCPU<3, Vector2c>;
extern template class mray::host::TextureCPU<3, Vector3c>;
extern template class mray::host::TextureCPU<3, Vector4c>;

extern template class mray::host::TextureCPU<3, uint16_t>;
extern template class mray::host::TextureCPU<3, Vector2us>;
extern template class mray::host::TextureCPU<3, Vector3us>;
extern template class mray::host::TextureCPU<3, Vector4us>;

extern template class mray::host::TextureCPU<3, int16_t>;
extern template class mray::host::TextureCPU<3, Vector2s>;
extern template class mray::host::TextureCPU<3, Vector3s>;
extern template class mray::host::TextureCPU<3, Vector4s>;