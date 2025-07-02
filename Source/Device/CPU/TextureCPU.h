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
class TextureCPU_Normal;

template<class T>
class TextureCPU_BC;

template<uint32_t D, class T>
class TextureCPU;

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

// These two classes are due to bug on MSVC
// https://stackoverflow.com/questions/68589314/how-to-define-a-specialized-class-method-outside-of-class-body-in-c
// (Clang also had this, but fixed)
// Issue here is that we cannot define the member functions outside of the
// class body.
// Instead of using enable_if, I've found the implementation above and used it.
// New MSVC versions do accept forward declaration of the base body so I've changed it.
//
// We drop the D parameter for BC texture since we only support 2D.
// (Is there event 3D file format that supports this?)
//
// TODO: In hindsight, %90 of the impl of BC and Normal Textures are the same.
// Concept guarded constructor / View() function could've been enough.
// so change it later.
template<uint32_t D, class T>
class TextureCPU_Normal
{
    public:
    static constexpr uint32_t ChannelCount  = VectorTypeToChannels<T>();
    static constexpr bool IsNormConvertible = ::IsNormConvertible<T>();
    static constexpr uint32_t Dims          = D;

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
    size_t  alignment   = MemAlloc::DefaultSystemAlignment();
    bool    allocated   = false;


    protected:
    public:
    // Constructors & Destructor
                        TextureCPU_Normal() = delete;
                        TextureCPU_Normal(const GPUDeviceCPU& device,
                                          const TextureInitParams<D>& p);
                        TextureCPU_Normal(const TextureCPU_Normal&) = delete;
                        TextureCPU_Normal(TextureCPU_Normal&&) noexcept;
    TextureCPU_Normal&  operator=(const TextureCPU_Normal&) = delete;
    TextureCPU_Normal&  operator=(TextureCPU_Normal&&) noexcept;
                        ~TextureCPU_Normal();

    // Direct view conversion (simple case)
    template<class QT>
    requires(std::is_same_v<QT, T>)
    TextureViewCPU<D, QT>  View() const;

    template<class QT>
    requires(!std::is_same_v<QT, T> &&
             (PixelTypeToChannels<T>() == PixelTypeToChannels<QT>()))
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

template<class T>
class TextureCPU_BC
{
    static constexpr uint32_t BC_TILE_SIZE  = 4;
    public:
    static constexpr uint32_t ChannelCount  = BCTypeToChannels<T>();
    static constexpr bool IsNormConvertible = true;
    static constexpr uint32_t Dims          = 2;
    static constexpr auto BlockSize         = BCTypeToBlockSize<T>();

    using Type              = T;
    using PaddedChannelType = Byte[BlockSize];

    private:
    const GPUDeviceCPU*     gpu;
    TextureInitParams<2>    texParams;
    // Allocation related
    size_t  size        = 0;
    size_t  alignment   = 0;
    bool    allocated = false;

    protected:
    public:
    // Constructors & Destructor
                    TextureCPU_BC() = delete;
                    TextureCPU_BC(const GPUDeviceCPU& device,
                                  const TextureInitParams<2>& p);
                    TextureCPU_BC(const TextureCPU_BC&) = delete;
                    TextureCPU_BC(TextureCPU_BC&&) noexcept = default;
    TextureCPU_BC&  operator=(const TextureCPU_BC&) = delete;
    TextureCPU_BC&  operator=(TextureCPU_BC&&) noexcept = default;
                    ~TextureCPU_BC() = default;

    // Only FloatX views are supported
    template<class QT>
    requires(!std::is_same_v<QT, T> &&
             (BCTypeToChannels<T>() == VectorTypeToChannels<QT>()))
    TextureViewCPU<2, QT>      View() const;

    size_t                  Size() const;
    size_t                  Alignment() const;

    TextureExtent<2>        Extents() const;
    uint32_t                MipCount() const;
    const GPUDeviceCPU&     Device() const;

    void                    CommitMemory(const GPUQueueCPU& queue,
                                         const TextureBackingMemoryCPU& deviceMem,
                                         size_t offset);
    void                    CopyFromAsync(const GPUQueueCPU& queue,
                                          uint32_t mipLevel,
                                          const TextureExtent<2>& offset,
                                          const TextureExtent<2>& size,
                                          Span<const PaddedChannelType> regionFrom);
    void                    CopyToAsync(Span<PaddedChannelType> regionFrom,
                                        const GPUQueueCPU& queue,
                                        uint32_t mipLevel,
                                        const TextureExtent<2>& offset,
                                        const TextureExtent<2>& sizes) const;
};

// This is the duplication part unfortunately
template<uint32_t D, NotBlockCompressedPixelC T>
class TextureCPU<D, T> : public TextureCPU_Normal<D, T>
{
    // Pull everything to this scope
    using Base = TextureCPU_Normal<D, T>;
    public:
    static constexpr uint32_t ChannelCount  = Base::ChannelCount;
    static constexpr bool IsNormConvertible = Base::IsNormConvertible;
    static constexpr uint32_t Dims          = Base::Dims;
    static constexpr bool IsBlockCompressed = false;
    using Type              = typename Base::Type;
    using PaddedChannelType = typename Base::PaddedChannelType;

    // Pull Constructor
    using Base::Base;
    // Functions were public so these are OK?
};

template<BlockCompressedPixelC T>
class TextureCPU<2, T> : public TextureCPU_BC<T>
{
    // Pull everything to this scope
    using Base = TextureCPU_BC<T>;
    public:
    static constexpr uint32_t ChannelCount  = Base::ChannelCount;
    static constexpr bool IsNormConvertible = Base::IsNormConvertible;
    static constexpr uint32_t Dims          = Base::Dims;
    static constexpr bool IsBlockCompressed = true;

    using Type              = typename Base::Type;
    using PaddedChannelType = typename Base::PaddedChannelType;

    // Pull Constructor
    using Base::Base;
    // Functions were public so these are OK?
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
extern template class mray::host::TextureCPU_Normal<1, Float>;
extern template class mray::host::TextureCPU_Normal<1, Vector2>;
extern template class mray::host::TextureCPU_Normal<1, Vector3>;
extern template class mray::host::TextureCPU_Normal<1, Vector4>;

extern template class mray::host::TextureCPU_Normal<1, uint8_t>;
extern template class mray::host::TextureCPU_Normal<1, Vector2uc>;
extern template class mray::host::TextureCPU_Normal<1, Vector3uc>;
extern template class mray::host::TextureCPU_Normal<1, Vector4uc>;

extern template class mray::host::TextureCPU_Normal<1, int8_t>;
extern template class mray::host::TextureCPU_Normal<1, Vector2c>;
extern template class mray::host::TextureCPU_Normal<1, Vector3c>;
extern template class mray::host::TextureCPU_Normal<1, Vector4c>;

extern template class mray::host::TextureCPU_Normal<1, uint16_t>;
extern template class mray::host::TextureCPU_Normal<1, Vector2us>;
extern template class mray::host::TextureCPU_Normal<1, Vector3us>;
extern template class mray::host::TextureCPU_Normal<1, Vector4us>;

extern template class mray::host::TextureCPU_Normal<1, int16_t>;
extern template class mray::host::TextureCPU_Normal<1, Vector2s>;
extern template class mray::host::TextureCPU_Normal<1, Vector3s>;
extern template class mray::host::TextureCPU_Normal<1, Vector4s>;

// Common Textures 2D
extern template class mray::host::TextureCPU_Normal<2, Float>;
extern template class mray::host::TextureCPU_Normal<2, Vector2>;
extern template class mray::host::TextureCPU_Normal<2, Vector3>;
extern template class mray::host::TextureCPU_Normal<2, Vector4>;

extern template class mray::host::TextureCPU_Normal<2, uint8_t>;
extern template class mray::host::TextureCPU_Normal<2, Vector2uc>;
extern template class mray::host::TextureCPU_Normal<2, Vector3uc>;
extern template class mray::host::TextureCPU_Normal<2, Vector4uc>;

extern template class mray::host::TextureCPU_Normal<2, int8_t>;
extern template class mray::host::TextureCPU_Normal<2, Vector2c>;
extern template class mray::host::TextureCPU_Normal<2, Vector3c>;
extern template class mray::host::TextureCPU_Normal<2, Vector4c>;

extern template class mray::host::TextureCPU_Normal<2, uint16_t>;
extern template class mray::host::TextureCPU_Normal<2, Vector2us>;
extern template class mray::host::TextureCPU_Normal<2, Vector3us>;
extern template class mray::host::TextureCPU_Normal<2, Vector4us>;

extern template class mray::host::TextureCPU_Normal<2, int16_t>;
extern template class mray::host::TextureCPU_Normal<2, Vector2s>;
extern template class mray::host::TextureCPU_Normal<2, Vector3s>;
extern template class mray::host::TextureCPU_Normal<2, Vector4s>;

extern template class mray::host::TextureCPU_BC<PixelBC1>;
extern template class mray::host::TextureCPU_BC<PixelBC2>;
extern template class mray::host::TextureCPU_BC<PixelBC3>;
extern template class mray::host::TextureCPU_BC<PixelBC4U>;
extern template class mray::host::TextureCPU_BC<PixelBC4S>;
extern template class mray::host::TextureCPU_BC<PixelBC5U>;
extern template class mray::host::TextureCPU_BC<PixelBC5S>;
extern template class mray::host::TextureCPU_BC<PixelBC6U>;
extern template class mray::host::TextureCPU_BC<PixelBC6S>;
extern template class mray::host::TextureCPU_BC<PixelBC7>;

// Common Textures 3D
extern template class mray::host::TextureCPU_Normal<3, Float>;
extern template class mray::host::TextureCPU_Normal<3, Vector2>;
extern template class mray::host::TextureCPU_Normal<3, Vector3>;
extern template class mray::host::TextureCPU_Normal<3, Vector4>;

extern template class mray::host::TextureCPU_Normal<3, uint8_t>;
extern template class mray::host::TextureCPU_Normal<3, Vector2uc>;
extern template class mray::host::TextureCPU_Normal<3, Vector3uc>;
extern template class mray::host::TextureCPU_Normal<3, Vector4uc>;

extern template class mray::host::TextureCPU_Normal<3, int8_t>;
extern template class mray::host::TextureCPU_Normal<3, Vector2c>;
extern template class mray::host::TextureCPU_Normal<3, Vector3c>;
extern template class mray::host::TextureCPU_Normal<3, Vector4c>;

extern template class mray::host::TextureCPU_Normal<3, uint16_t>;
extern template class mray::host::TextureCPU_Normal<3, Vector2us>;
extern template class mray::host::TextureCPU_Normal<3, Vector3us>;
extern template class mray::host::TextureCPU_Normal<3, Vector4us>;

extern template class mray::host::TextureCPU_Normal<3, int16_t>;
extern template class mray::host::TextureCPU_Normal<3, Vector2s>;
extern template class mray::host::TextureCPU_Normal<3, Vector3s>;
extern template class mray::host::TextureCPU_Normal<3, Vector4s>;

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

extern template class mray::host::TextureCPU<2, PixelBC1>;
extern template class mray::host::TextureCPU<2, PixelBC2>;
extern template class mray::host::TextureCPU<2, PixelBC3>;
extern template class mray::host::TextureCPU<2, PixelBC4U>;
extern template class mray::host::TextureCPU<2, PixelBC4S>;
extern template class mray::host::TextureCPU<2, PixelBC5U>;
extern template class mray::host::TextureCPU<2, PixelBC5S>;
extern template class mray::host::TextureCPU<2, PixelBC6U>;
extern template class mray::host::TextureCPU<2, PixelBC6S>;
extern template class mray::host::TextureCPU<2, PixelBC7>;

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