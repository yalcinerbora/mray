#include "TextureMemory.h"

// Now the hard part
// Actual impl. of common texture
namespace TexDetail
{
template <typename T>
class Concept : public CommonTextureI
{
    private:
    T tex;

    public:
    template<class... Args>
                        Concept(Args&&...);

    void                CommitMemory(const GPUQueue& queue,
                                        const TextureBackingMemory& deviceMem,
                                        size_t offset) override;
    size_t              Size() const override;
    size_t              Alignment() const override;
    uint32_t            MipCount() const override;
    //
    TextureExtent<3>    Extents() const override;
    uint32_t            DimensionCount() const;
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      TransientData regionFrom) override;
    GenericTextureView  View() const override;
};

}

template<size_t StorageSize, size_t Alignment>
class alignas(Alignment) CommonTextureT
{
    private:
    std::array<Byte, StorageSize> storage;
    MRayColorSpaceEnum  colorSpace;
    AttributeIsColor    isColor;

    CommonTextureI*         Impl();
    const CommonTextureI*   Impl() const;

    public:
    template<class T, class... Args>
                CommonTextureT(std::in_place_type_t<T>,
                               MRayColorSpaceEnum, AttributeIsColor,
                               Args&&...);

    void        CommitMemory(const GPUQueue& queue,
                             const TextureBackingMemory& deviceMem,
                             size_t offset);
    size_t      Size() const;
    size_t      Alignment() const;
    uint32_t    MipCount() const;
    //
    TextureExtent<3>    Extents() const;
    uint32_t            DimensionCount() const;
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      TransientData regionFrom);
    GenericTextureView  View() const;

    // These are extra functionality that will be built on top of
    MRayColorSpaceEnum  ColorSpace() const;
    AttributeIsColor    IsColor() const;

};

namespace TexDetail
{

template<class T>
template<class... Args>
Concept<T>::Concept(Args&&... args)
    : tex(std::forward<Args>(args)...)
{}

template<class T>
void Concept<T>::CommitMemory(const GPUQueue& queue,
                              const TextureBackingMemory& deviceMem,
                              size_t offset)
{
    tex.CommitMemory(queue, deviceMem, offset);
}

template<class T>
size_t Concept<T>::Size() const
{
    return tex.Size();
}

template<class T>
size_t Concept<T>::Alignment() const
{
    return tex.Alignment();
}

template<class T>
uint32_t Concept<T>::MipCount() const
{
    return tex.MipCount();
}

template<class T>
TextureExtent<3> Concept<T>::Extents() const
{
    auto ext = tex.Extents();
    if constexpr(T::Dims == 1)
        return TextureExtent<3>(ext, 0, 0);
    else if constexpr(T::Dims == 2)
        return TextureExtent<3>(ext[0], ext[1], 0);
    else return ext;
}

template<class T>
uint32_t Concept<T>::DimensionCount() const
{
    return T::Dims;
}

template<class T>
void Concept<T>::CopyFromAsync(const GPUQueue& queue,
                               uint32_t mipLevel,
                               const TextureExtent<3>& offset,
                               const TextureExtent<3>& size,
                               TransientData regionFrom)
{
    using PixelType = typename T::Type;
    using ExtType = TextureExtent<T::Dims>;
    ExtType offsetIn;
    ExtType sizeIn;

    auto ext = tex.Extents();
    if constexpr(T::Dims == 1)
    {
        offsetIn = ExtType(offset);
        sizeIn = ExtType(size);
    }
    else
    {
        offsetIn = ExtType(offset);
        sizeIn = ExtType(size);
    }

    Span<const PixelType> input = regionFrom.AccessAs<const PixelType>();
    tex.CopyFromAsync(queue, mipLevel, offsetIn, sizeIn, input);
}

template<class T>
GenericTextureView Concept<T>::View() const
{
    static constexpr uint32_t ChannelCount = T::PaddedChannelType::Dims;

    if constexpr(ChannelCount == 1)
        return tex.View<Float>();
    else if constexpr(ChannelCount == 2)
        return tex.View<Vector2>();
    else
        return tex.View<Vector3>();
}

}

template<size_t S, size_t A>
inline
CommonTextureI* CommonTextureT<S,A>::Impl()
{
    // TODO: Are these correct?
    return std::launder(reinterpret_cast<CommonTextureI*>(storage.data()));
}

template<size_t S, size_t A>
inline
const CommonTextureI* CommonTextureT<S, A>::Impl() const
{
    // TODO: Are these correct?
   return std::launder(reinterpret_cast<const CommonTextureI*>(storage.data()));
}

template<size_t S, size_t A>
template<class T, class... Args>
inline
CommonTextureT<S, A>::CommonTextureT(std::in_place_type_t<T>,
                                     MRayColorSpaceEnum cs, AttributeIsColor col,
                                     Args&&... args)
    : colorSpace(cs)
    , isColor(col)
{
    static_assert(sizeof(T) <= S, "Unable construct type over storage!");
    T* ptr = reinterpret_cast<T*>(storage.data());
    std::construct_at(ptr, std::forward<Args>(args)...);
}

template<size_t S, size_t A>
inline
void CommonTextureT<S, A>::CommitMemory(const GPUQueue& queue,
                                       const TextureBackingMemory& deviceMem,
                                       size_t offset)
{
    Impl()->CommitMemory(queue, deviceMem, offset);
}

template<size_t S, size_t A>
inline
size_t CommonTextureT<S, A>::Size() const
{
    return Impl()->Size();
}

template<size_t S, size_t A>
inline
size_t CommonTextureT<S, A>::Alignment() const
{
    return Impl()->Alignment();
}

template<size_t S, size_t A>
inline
uint32_t CommonTextureT<S, A>::MipCount() const
{
    return Impl()->Size();
}

template<size_t S, size_t A>
inline
TextureExtent<3> CommonTextureT<S, A>::Extents() const
{
    Impl()->Size();
}

template<size_t S, size_t A>
inline
uint32_t CommonTextureT<S, A>::DimensionCount() const
{
    Impl()->DimensionCount();
}

template<size_t S, size_t A>
inline
void CommonTextureT<S, A>::CopyFromAsync(const GPUQueue& queue,
                                        uint32_t mipLevel,
                                        const TextureExtent<3>& offset,
                                        const TextureExtent<3>& size,
                                        TransientData regionFrom)
{
    Impl()->DimensionCount(queue, mipLevel,
                           offset, size,
                           regionFrom);
}

template<size_t S, size_t A>
inline
GenericTextureView CommonTextureT<S, A>::View() const
{
    return Impl()->View();
}

using CommonTexture = CommonTextureT<std::max(sizeof(Texture<3, Vector4>),
                                              sizeof(Texture<2, Vector4>)),
                                     std::max(alignof(Texture<3, Vector4>),
                                              alignof(Texture<2, Vector4>))>;

TextureMemory::TextureMemory(const GPUSystem& sys)
    : gpuSystem(sys)
{}

TextureId TextureMemory::CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                         MRayPixelTypeRT pixType,
                                         AttributeIsColor isColor)
{
    MRayColorSpaceEnum colorSpace = MRayColorSpaceEnum::MR_DEFAULT;
    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& queue = device.GetQueue(0);

    TextureInitParams<2> p;
    p.size = size;
    p.mipCount = mipCount;
    CommonTexture tex = std::visit([&](auto&& v) -> CommonTexture
    {
        using enum MRayPixelEnum;
        using ArgType = std::remove_cvref_t<decltype(v)>;
        if constexpr(!IsBlockCompressedType<ArgType>)
        {
            using Type = typename ArgType::Type;
            return CommonTexture(std::in_place_type_t<Texture<2, Type>>{},
                                 colorSpace, isColor,
                                 device, p);
        }
        else throw MRayError("Block compressed types are not supported!");
    }, pixType);

    TextureId id = TextureId(texCounter.fetch_add(1));
    //textures.emplace(id, std::move(tex));

    return id;
}