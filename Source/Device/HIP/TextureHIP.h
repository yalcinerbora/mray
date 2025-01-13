#pragma once

#include "DeviceMemoryHIP.h"
#include "TextureViewHIP.h"
#include "DefinitionsHIP.h"
#include "GPUSystemHIP.h"

#include "../GPUTypes.h"

#include "Core/Definitions.h"

namespace mray::hip
{

using BCEnumFinder = TypeFinder::T_VMapper:: template Map
<
    // TODO: BC is not supported in HIP yet, change these later
    TypeFinder::T_VMapper::template TVPair<PixelBC1,  hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC2,  hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC3,  hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC4U, hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC4S, hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC5U, hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC5S, hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC6U, hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC6S, hipChannelFormatKindUnsigned>,
    TypeFinder::T_VMapper::template TVPair<PixelBC7,  hipChannelFormatKindUnsigned>
>;

class TextureBackingMemoryHIP;

template <class T>
constexpr bool IsNormConvertibleHIP()
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

template<uint32_t D, class T>
class TextureHIP_Normal;

template<class T>
class TextureHIP_BC;

template<uint32_t D, class T>
class TextureHIP;

// RAII wrapper for the surface object
template<uint32_t D, class T>
class RWTextureRefHIP
{
    friend class TextureHIP_Normal<D, T>;

    private:
    hipSurfaceObject_t     s = hipSurfaceObject_t(0);
    // Hide this, only texture class can create
                            RWTextureRefHIP(hipSurfaceObject_t);
    public:
    // Constructors & Destructor
                            RWTextureRefHIP(const RWTextureRefHIP&) = delete;
                            RWTextureRefHIP(RWTextureRefHIP&&);
    RWTextureRefHIP&        operator=(const RWTextureRefHIP&) = delete;
    RWTextureRefHIP&        operator=(RWTextureRefHIP&&);
                            ~RWTextureRefHIP();
    //
    RWTextureViewHIP<D, T>  View() const;
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
class TextureHIP_Normal
{
    public:
    static constexpr uint32_t ChannelCount  = VectorTypeToChannels<T>();
    static constexpr bool IsNormConvertible = IsNormConvertibleHIP<T>();
    static constexpr uint32_t Dims          = D;

    using Type              = T;
    using HipType           = typename VectorTypeToHIP::template Find<T>;
    using PaddedChannelType = PaddedChannel<ChannelCount, T>;

    // Sanity Check
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");

    private:
    const GPUDeviceHIP*     gpu;
    hipTextureObject_t      tex         = hipTextureObject_t(0);
    hipMipmappedArray_t     data        = nullptr;
    TextureInitParams<D>    texParams;

    // Allocation related
    size_t  size        = 0;
    size_t  alignment   = 0;
    bool    allocated   = false;

    protected:
    public:
    // Constructors & Destructor
                        TextureHIP_Normal() = delete;
                        TextureHIP_Normal(const GPUDeviceHIP& device,
                                           const TextureInitParams<D>& p);
                        TextureHIP_Normal(const TextureHIP_Normal&) = delete;
                        TextureHIP_Normal(TextureHIP_Normal&&) noexcept;
    TextureHIP_Normal& operator=(const TextureHIP_Normal&) = delete;
    TextureHIP_Normal& operator=(TextureHIP_Normal&&) noexcept;
                        ~TextureHIP_Normal();

    // Direct view conversion (simple case)
    template<class QT> requires(std::is_same_v<QT, T>)
    TextureViewHIP<D, QT>  View() const;

    template<class QT>
    requires(!std::is_same_v<QT, T> &&
             (VectorTypeToChannels<T>() ==
              VectorTypeToChannels<QT>()))
    TextureViewHIP<D, QT>   View() const;

    RWTextureRefHIP<D, T>   GenerateRWRef(uint32_t mipLevel);

    size_t                  Size() const;
    size_t                  Alignment() const;

    TextureExtent<D>        Extents() const;
    uint32_t                MipCount() const;
    const GPUDeviceHIP&     Device() const;

    void                    CommitMemory(const GPUQueueHIP& queue,
                                         const TextureBackingMemoryHIP& deviceMem,
                                         size_t offset);
    void                    CopyFromAsync(const GPUQueueHIP& queue,
                                          uint32_t mipLevel,
                                          const TextureExtent<D>& offset,
                                          const TextureExtent<D>& size,
                                          Span<const PaddedChannelType> regionFrom);
};

template<class T>
class TextureHIP_BC
{
    static constexpr uint32_t BC_TILE_SIZE  = 4;
    public:
    static constexpr uint32_t ChannelCount  = BCTypeToChannels<T>();
    static constexpr bool IsNormConvertible = true;
    static constexpr uint32_t Dims          = 2;
    static constexpr auto HipTypeEnum       = static_cast<hipChannelFormatKind>(BCEnumFinder::Find<T>);
    static constexpr auto BlockSize         = BCTypeToBlockSize<T>();

    using Type              = T;
    using PaddedChannelType = Byte[BlockSize];

    private:
    const GPUDeviceHIP*     gpu;
    hipTextureObject_t      tex         = hipTextureObject_t(0);
    hipMipmappedArray_t     data        = nullptr;
    TextureInitParams<2>    texParams;
    // Allocation related
    size_t  size        = 0;
    size_t  alignment   = 0;
    bool    allocated = false;

    protected:
    public:
    // Constructors & Destructor
                    TextureHIP_BC() = delete;
                    TextureHIP_BC(const GPUDeviceHIP& device,
                                   const TextureInitParams<2>& p);
                    TextureHIP_BC(const TextureHIP_BC&) = delete;
                    TextureHIP_BC(TextureHIP_BC&&) noexcept;
    TextureHIP_BC&  operator=(const TextureHIP_BC&) = delete;
    TextureHIP_BC&  operator=(TextureHIP_BC&&) noexcept;
                    ~TextureHIP_BC();

    // Only FloatX views are supported
    template<class QT>
    requires(!std::is_same_v<QT, T> &&
             (BCTypeToChannels<T>() == VectorTypeToChannels<QT>()))
    TextureViewHIP<2, QT>   View() const;

    size_t                  Size() const;
    size_t                  Alignment() const;

    TextureExtent<2>        Extents() const;
    uint32_t                MipCount() const;
    const GPUDeviceHIP&     Device() const;

    void                    CommitMemory(const GPUQueueHIP& queue,
                                         const TextureBackingMemoryHIP& deviceMem,
                                         size_t offset);
    void                    CopyFromAsync(const GPUQueueHIP& queue,
                                          uint32_t mipLevel,
                                          const TextureExtent<2>& offset,
                                          const TextureExtent<2>& size,
                                          Span<const PaddedChannelType> regionFrom);
    void                    CopyToAsync(Span<PaddedChannelType> regionFrom,
                                        const GPUQueueHIP& queue,
                                        uint32_t mipLevel,
                                        const TextureExtent<2>& offset,
                                        const TextureExtent<2>& sizes) const;
};

// This is the duplication part unfortunately
template<uint32_t D, NotBlockCompressedPixelC T>
class TextureHIP<D, T> : public TextureHIP_Normal<D, T>
{
    // Pull everything to this scope
    using Base = TextureHIP_Normal<D, T>;
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
class TextureHIP<2, T> : public TextureHIP_BC<T>
{
    // Pull everything to this scope
    using Base = TextureHIP_BC<T>;
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

class TextureBackingMemoryHIP
{
    MRAY_HYBRID
    friend hipMemGenericAllocationHandle_t ToHandleHIP(const TextureBackingMemoryHIP& mem);

    private:
    const GPUDeviceHIP*             gpu;
    size_t                          size;
    size_t                          allocSize;
    hipMemGenericAllocationHandle_t memHandle;

    public:
    // Constructors & Destructor
                                TextureBackingMemoryHIP(const GPUDeviceHIP& device);
                                TextureBackingMemoryHIP(const GPUDeviceHIP& device, size_t sizeInBytes);
                                TextureBackingMemoryHIP(const TextureBackingMemoryHIP&) = delete;
                                TextureBackingMemoryHIP(TextureBackingMemoryHIP&&) noexcept;
                                ~TextureBackingMemoryHIP();
    TextureBackingMemoryHIP&    operator=(const TextureBackingMemoryHIP&) = delete;
    TextureBackingMemoryHIP&    operator=(TextureBackingMemoryHIP&&) noexcept;

    void                        ResizeBuffer(size_t newSize);
    const GPUDeviceHIP&         Device() const;
    size_t                      Size() const;
};

MRAY_HYBRID
inline hipMemGenericAllocationHandle_t ToHandleHIP(const TextureBackingMemoryHIP& mem)
{
    return mem.memHandle;
}

}

#include "TextureHIP.hpp"

// Common Textures 1D
extern template class mray::hip::TextureHIP_Normal<1, Float>;
extern template class mray::hip::TextureHIP_Normal<1, Vector2>;
extern template class mray::hip::TextureHIP_Normal<1, Vector3>;
extern template class mray::hip::TextureHIP_Normal<1, Vector4>;

extern template class mray::hip::TextureHIP_Normal<1, uint8_t>;
extern template class mray::hip::TextureHIP_Normal<1, Vector2uc>;
extern template class mray::hip::TextureHIP_Normal<1, Vector3uc>;
extern template class mray::hip::TextureHIP_Normal<1, Vector4uc>;

extern template class mray::hip::TextureHIP_Normal<1, int8_t>;
extern template class mray::hip::TextureHIP_Normal<1, Vector2c>;
extern template class mray::hip::TextureHIP_Normal<1, Vector3c>;
extern template class mray::hip::TextureHIP_Normal<1, Vector4c>;

extern template class mray::hip::TextureHIP_Normal<1, uint16_t>;
extern template class mray::hip::TextureHIP_Normal<1, Vector2us>;
extern template class mray::hip::TextureHIP_Normal<1, Vector3us>;
extern template class mray::hip::TextureHIP_Normal<1, Vector4us>;

extern template class mray::hip::TextureHIP_Normal<1, int16_t>;
extern template class mray::hip::TextureHIP_Normal<1, Vector2s>;
extern template class mray::hip::TextureHIP_Normal<1, Vector3s>;
extern template class mray::hip::TextureHIP_Normal<1, Vector4s>;

// Common Textures 2D
extern template class mray::hip::TextureHIP_Normal<2, Float>;
extern template class mray::hip::TextureHIP_Normal<2, Vector2>;
extern template class mray::hip::TextureHIP_Normal<2, Vector3>;
extern template class mray::hip::TextureHIP_Normal<2, Vector4>;

extern template class mray::hip::TextureHIP_Normal<2, uint8_t>;
extern template class mray::hip::TextureHIP_Normal<2, Vector2uc>;
extern template class mray::hip::TextureHIP_Normal<2, Vector3uc>;
extern template class mray::hip::TextureHIP_Normal<2, Vector4uc>;

extern template class mray::hip::TextureHIP_Normal<2, int8_t>;
extern template class mray::hip::TextureHIP_Normal<2, Vector2c>;
extern template class mray::hip::TextureHIP_Normal<2, Vector3c>;
extern template class mray::hip::TextureHIP_Normal<2, Vector4c>;

extern template class mray::hip::TextureHIP_Normal<2, uint16_t>;
extern template class mray::hip::TextureHIP_Normal<2, Vector2us>;
extern template class mray::hip::TextureHIP_Normal<2, Vector3us>;
extern template class mray::hip::TextureHIP_Normal<2, Vector4us>;

extern template class mray::hip::TextureHIP_Normal<2, int16_t>;
extern template class mray::hip::TextureHIP_Normal<2, Vector2s>;
extern template class mray::hip::TextureHIP_Normal<2, Vector3s>;
extern template class mray::hip::TextureHIP_Normal<2, Vector4s>;

extern template class mray::hip::TextureHIP_BC<PixelBC1>;
extern template class mray::hip::TextureHIP_BC<PixelBC2>;
extern template class mray::hip::TextureHIP_BC<PixelBC3>;
extern template class mray::hip::TextureHIP_BC<PixelBC4U>;
extern template class mray::hip::TextureHIP_BC<PixelBC4S>;
extern template class mray::hip::TextureHIP_BC<PixelBC5U>;
extern template class mray::hip::TextureHIP_BC<PixelBC5S>;
extern template class mray::hip::TextureHIP_BC<PixelBC6U>;
extern template class mray::hip::TextureHIP_BC<PixelBC6S>;
extern template class mray::hip::TextureHIP_BC<PixelBC7>;

// Common Textures 3D
extern template class mray::hip::TextureHIP_Normal<3, Float>;
extern template class mray::hip::TextureHIP_Normal<3, Vector2>;
extern template class mray::hip::TextureHIP_Normal<3, Vector3>;
extern template class mray::hip::TextureHIP_Normal<3, Vector4>;

extern template class mray::hip::TextureHIP_Normal<3, uint8_t>;
extern template class mray::hip::TextureHIP_Normal<3, Vector2uc>;
extern template class mray::hip::TextureHIP_Normal<3, Vector3uc>;
extern template class mray::hip::TextureHIP_Normal<3, Vector4uc>;

extern template class mray::hip::TextureHIP_Normal<3, int8_t>;
extern template class mray::hip::TextureHIP_Normal<3, Vector2c>;
extern template class mray::hip::TextureHIP_Normal<3, Vector3c>;
extern template class mray::hip::TextureHIP_Normal<3, Vector4c>;

extern template class mray::hip::TextureHIP_Normal<3, uint16_t>;
extern template class mray::hip::TextureHIP_Normal<3, Vector2us>;
extern template class mray::hip::TextureHIP_Normal<3, Vector3us>;
extern template class mray::hip::TextureHIP_Normal<3, Vector4us>;

extern template class mray::hip::TextureHIP_Normal<3, int16_t>;
extern template class mray::hip::TextureHIP_Normal<3, Vector2s>;
extern template class mray::hip::TextureHIP_Normal<3, Vector3s>;
extern template class mray::hip::TextureHIP_Normal<3, Vector4s>;


// Common Textures 1D
extern template class mray::hip::TextureHIP<1, Float>;
extern template class mray::hip::TextureHIP<1, Vector2>;
extern template class mray::hip::TextureHIP<1, Vector3>;
extern template class mray::hip::TextureHIP<1, Vector4>;

extern template class mray::hip::TextureHIP<1, uint8_t>;
extern template class mray::hip::TextureHIP<1, Vector2uc>;
extern template class mray::hip::TextureHIP<1, Vector3uc>;
extern template class mray::hip::TextureHIP<1, Vector4uc>;

extern template class mray::hip::TextureHIP<1, int8_t>;
extern template class mray::hip::TextureHIP<1, Vector2c>;
extern template class mray::hip::TextureHIP<1, Vector3c>;
extern template class mray::hip::TextureHIP<1, Vector4c>;

extern template class mray::hip::TextureHIP<1, uint16_t>;
extern template class mray::hip::TextureHIP<1, Vector2us>;
extern template class mray::hip::TextureHIP<1, Vector3us>;
extern template class mray::hip::TextureHIP<1, Vector4us>;

extern template class mray::hip::TextureHIP<1, int16_t>;
extern template class mray::hip::TextureHIP<1, Vector2s>;
extern template class mray::hip::TextureHIP<1, Vector3s>;
extern template class mray::hip::TextureHIP<1, Vector4s>;

// Common Textures 2D
extern template class mray::hip::TextureHIP<2, Float>;
extern template class mray::hip::TextureHIP<2, Vector2>;
extern template class mray::hip::TextureHIP<2, Vector3>;
extern template class mray::hip::TextureHIP<2, Vector4>;

extern template class mray::hip::TextureHIP<2, uint8_t>;
extern template class mray::hip::TextureHIP<2, Vector2uc>;
extern template class mray::hip::TextureHIP<2, Vector3uc>;
extern template class mray::hip::TextureHIP<2, Vector4uc>;

extern template class mray::hip::TextureHIP<2, int8_t>;
extern template class mray::hip::TextureHIP<2, Vector2c>;
extern template class mray::hip::TextureHIP<2, Vector3c>;
extern template class mray::hip::TextureHIP<2, Vector4c>;

extern template class mray::hip::TextureHIP<2, uint16_t>;
extern template class mray::hip::TextureHIP<2, Vector2us>;
extern template class mray::hip::TextureHIP<2, Vector3us>;
extern template class mray::hip::TextureHIP<2, Vector4us>;

extern template class mray::hip::TextureHIP<2, int16_t>;
extern template class mray::hip::TextureHIP<2, Vector2s>;
extern template class mray::hip::TextureHIP<2, Vector3s>;
extern template class mray::hip::TextureHIP<2, Vector4s>;

extern template class mray::hip::TextureHIP<2, PixelBC1>;
extern template class mray::hip::TextureHIP<2, PixelBC2>;
extern template class mray::hip::TextureHIP<2, PixelBC3>;
extern template class mray::hip::TextureHIP<2, PixelBC4U>;
extern template class mray::hip::TextureHIP<2, PixelBC4S>;
extern template class mray::hip::TextureHIP<2, PixelBC5U>;
extern template class mray::hip::TextureHIP<2, PixelBC5S>;
extern template class mray::hip::TextureHIP<2, PixelBC6U>;
extern template class mray::hip::TextureHIP<2, PixelBC6S>;
extern template class mray::hip::TextureHIP<2, PixelBC7>;

// Common Textures 3D
extern template class mray::hip::TextureHIP<3, Float>;
extern template class mray::hip::TextureHIP<3, Vector2>;
extern template class mray::hip::TextureHIP<3, Vector3>;
extern template class mray::hip::TextureHIP<3, Vector4>;

extern template class mray::hip::TextureHIP<3, uint8_t>;
extern template class mray::hip::TextureHIP<3, Vector2uc>;
extern template class mray::hip::TextureHIP<3, Vector3uc>;
extern template class mray::hip::TextureHIP<3, Vector4uc>;

extern template class mray::hip::TextureHIP<3, int8_t>;
extern template class mray::hip::TextureHIP<3, Vector2c>;
extern template class mray::hip::TextureHIP<3, Vector3c>;
extern template class mray::hip::TextureHIP<3, Vector4c>;

extern template class mray::hip::TextureHIP<3, uint16_t>;
extern template class mray::hip::TextureHIP<3, Vector2us>;
extern template class mray::hip::TextureHIP<3, Vector3us>;
extern template class mray::hip::TextureHIP<3, Vector4us>;

extern template class mray::hip::TextureHIP<3, int16_t>;
extern template class mray::hip::TextureHIP<3, Vector2s>;
extern template class mray::hip::TextureHIP<3, Vector3s>;
extern template class mray::hip::TextureHIP<3, Vector4s>;