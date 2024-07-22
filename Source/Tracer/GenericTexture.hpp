#pragma once

template<size_t S, size_t A>
inline
GenericTextureI* GenericTextureT<S,A>::Impl()
{
    // TODO: Are these correct?
    GenericTextureI* ptr = reinterpret_cast<GenericTextureI*>(storage.data());
    return std::launder(ptr);
}

template<size_t S, size_t A>
inline
const GenericTextureI* GenericTextureT<S, A>::Impl() const
{
    // TODO: Are these correct?
    const GenericTextureI* ptr = reinterpret_cast<const GenericTextureI*>(storage.data());
    return std::launder(ptr);
}

template<size_t S, size_t A>
template<class T, class... Args>
inline
GenericTextureT<S, A>::GenericTextureT(std::in_place_type_t<T>,
                                     MRayColorSpaceEnum cs, Float gammaIn,
                                     AttributeIsColor col, MRayPixelTypeRT pt,
                                     Args&&... args)
    : colorSpace(cs)
    , gamma(gammaIn)
    , isColor(col)
    , pixelType(pt)
{
    isMipLoaded.Reset();
    using ConceptType = TexDetail::Concept<T>;
    static_assert(sizeof(ConceptType) <= S, "Unable construct type over storage!");
    ConceptType* ptr = reinterpret_cast<ConceptType*>(storage.data());
    impl = std::construct_at(ptr, std::forward<Args>(args)...);
}

template<size_t S, size_t A>
GenericTextureT<S, A>::~GenericTextureT()
{
    std::destroy_at(Impl());
}

template<size_t S, size_t A>
inline
void GenericTextureT<S, A>::CommitMemory(const GPUQueue& queue,
                                       const TextureBackingMemory& deviceMem,
                                       size_t offset)
{
    Impl()->CommitMemory(queue, deviceMem, offset);
}

template<size_t S, size_t A>
inline
size_t GenericTextureT<S, A>::Size() const
{
    return Impl()->Size();
}

template<size_t S, size_t A>
inline
size_t GenericTextureT<S, A>::Alignment() const
{
    return Impl()->Alignment();
}

template<size_t S, size_t A>
inline
uint32_t GenericTextureT<S, A>::MipCount() const
{
    return Impl()->MipCount();
}

template<size_t S, size_t A>
inline
TextureExtent<3> GenericTextureT<S, A>::Extents() const
{
    return Impl()->Extents();
}

template<size_t S, size_t A>
inline
uint32_t GenericTextureT<S, A>::DimensionCount() const
{
    return Impl()->DimensionCount();
}

template<size_t S, size_t A>
inline
void GenericTextureT<S, A>::CopyFromAsync(const GPUQueue& queue,
                                        uint32_t mipLevel,
                                        const TextureExtent<3>& offset,
                                        const TextureExtent<3>& size,
                                        TransientData regionFrom)
{
    isMipLoaded[mipLevel] = true;
    Impl()->CopyFromAsync(queue, mipLevel,
                          offset, size,
                          std::move(regionFrom));
}

template<size_t S, size_t A>
inline
void GenericTextureT<S, A>::CopyFromAsync(const GPUQueue& queue,
                                          uint32_t mipLevel,
                                          const TextureExtent<3>& offset,
                                          const TextureExtent<3>& size,
                                          Span<const Byte> regionFrom)
{
    isMipLoaded[mipLevel] = true;
    Impl()->CopyFromAsync(queue, mipLevel,
                          offset, size,
                          regionFrom);
}

template<size_t S, size_t A>
inline
GenericTextureView GenericTextureT<S, A>::View(TextureReadMode mode) const
{
    return Impl()->View(mode);
}

template<size_t S, size_t A>
inline
bool GenericTextureT<S, A>::HasRWView() const
{
    return Impl()->HasRWView();
}

template<size_t S, size_t A>
inline
SurfRefVariant GenericTextureT<S, A>::RWView(uint32_t mipLevel)
{
    return Impl()->RWView(mipLevel);
}

template<size_t S, size_t A>
inline
const GPUDevice& GenericTextureT<S, A>::Device() const
{
    return Impl()->Device();
}

template<size_t S, size_t A>
inline
uint32_t GenericTextureT<S, A>::ChannelCount() const
{
    return Impl()->ChannelCount();
}

template<size_t S, size_t A>
inline typename GenericTextureT<S, A>::MipIsLoadedBits
GenericTextureT<S, A>::ValidMips() const
{
    return isMipLoaded;
}

template<size_t S, size_t A>
inline
MRayColorSpaceEnum GenericTextureT<S, A>::ColorSpace() const
{
    return colorSpace;
}

template<size_t S, size_t A>
inline
Float GenericTextureT<S, A>::Gamma() const
{
    return gamma;
}

template<size_t S, size_t A>
inline
AttributeIsColor GenericTextureT<S, A>::IsColor() const
{
    return isColor;
}

template<size_t S, size_t A>
inline
MRayPixelTypeRT GenericTextureT<S, A>::PixelType() const
{
    return pixelType;
}

template<size_t S, size_t A>
inline
void GenericTextureT<S, A>::SetAllMipsToLoaded()
{
    isMipLoaded.Set();
}

template<size_t S, size_t A>
inline
void GenericTextureT<S, A>::SetColorSpace(MRayColorSpaceEnum e, Float gammaIn)
{
    colorSpace = e;
    gamma = gammaIn;
}