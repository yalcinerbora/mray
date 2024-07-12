#pragma once

template<size_t S, size_t A>
inline
CommonTextureI* CommonTextureT<S,A>::Impl()
{
    // TODO: Are these correct?
    CommonTextureI* ptr = reinterpret_cast<CommonTextureI*>(storage.data());
    return std::launder(ptr);
}

template<size_t S, size_t A>
inline
const CommonTextureI* CommonTextureT<S, A>::Impl() const
{
    // TODO: Are these correct?
    const CommonTextureI* ptr = reinterpret_cast<const CommonTextureI*>(storage.data());
    return std::launder(ptr);
}

template<size_t S, size_t A>
template<class T, class... Args>
inline
CommonTextureT<S, A>::CommonTextureT(std::in_place_type_t<T>,
                                     MRayColorSpaceEnum cs, AttributeIsColor col,
                                     MRayPixelTypeRT pt,
                                     Args&&... args)
    : colorSpace(cs)
    , isColor(col)
    , pixelType(pt)
{
    using ConceptType = TexDetail::Concept<T>;
    static_assert(sizeof(ConceptType) <= S, "Unable construct type over storage!");
    ConceptType* ptr = reinterpret_cast<ConceptType*>(storage.data());
    impl = std::construct_at(ptr, std::forward<Args>(args)...);
}

template<size_t S, size_t A>
CommonTextureT<S, A>::~CommonTextureT()
{
    std::destroy_at(Impl());
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
    return Impl()->MipCount();
}

template<size_t S, size_t A>
inline
TextureExtent<3> CommonTextureT<S, A>::Extents() const
{
    return Impl()->Extents();
}

template<size_t S, size_t A>
inline
uint32_t CommonTextureT<S, A>::DimensionCount() const
{
    return Impl()->DimensionCount();
}

template<size_t S, size_t A>
inline
void CommonTextureT<S, A>::CopyFromAsync(const GPUQueue& queue,
                                        uint32_t mipLevel,
                                        const TextureExtent<3>& offset,
                                        const TextureExtent<3>& size,
                                        TransientData regionFrom)
{
    Impl()->CopyFromAsync(queue, mipLevel,
                          offset, size,
                          std::move(regionFrom));
}

template<size_t S, size_t A>
inline
GenericTextureView CommonTextureT<S, A>::View() const
{
    return Impl()->View();
}

template<size_t S, size_t A>
inline
const GPUDevice& CommonTextureT<S, A>::Device() const
{
    return Impl()->Device();
}

template<size_t S, size_t A>
inline
uint32_t CommonTextureT<S, A>::ChannelCount() const
{
    return Impl()->ChannelCount();
}

template<size_t S, size_t A>
inline
MRayColorSpaceEnum CommonTextureT<S, A>::ColorSpace() const
{
    return colorSpace;
}

template<size_t S, size_t A>
inline
AttributeIsColor CommonTextureT<S, A>::IsColor() const
{
    return isColor;
}

template<size_t S, size_t A>
inline
MRayPixelTypeRT CommonTextureT<S, A>::PixelType() const
{
    return pixelType;
}
