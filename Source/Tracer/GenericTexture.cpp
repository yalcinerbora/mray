#include "TextureMemory.h"

GenericTextureI* GenericTextureT::Impl()
{
    // TODO: Are these correct?
    GenericTextureI* ptr = reinterpret_cast<GenericTextureI*>(storage.data());
    return std::launder(ptr);
}

const GenericTextureI* GenericTextureT::Impl() const
{
    // TODO: Are these correct?
    const GenericTextureI* ptr = reinterpret_cast<const GenericTextureI*>(storage.data());
    return std::launder(ptr);
}

GenericTextureT::~GenericTextureT()
{
    std::destroy_at(Impl());
}

void GenericTextureT::CommitMemory(const GPUQueue& queue,
                                   const TextureBackingMemory& deviceMem,
                                   size_t offset)
{
    Impl()->CommitMemory(queue, deviceMem, offset);
}

size_t GenericTextureT::Size() const
{
    return Impl()->Size();
}

size_t GenericTextureT::Alignment() const
{
    return Impl()->Alignment();
}

uint32_t GenericTextureT::MipCount() const
{
    return Impl()->MipCount();
}

TextureExtent<3> GenericTextureT::Extents() const
{
    return Impl()->Extents();
}

uint32_t GenericTextureT::DimensionCount() const
{
    return Impl()->DimensionCount();
}

void GenericTextureT::CopyFromAsync(const GPUQueue& queue,
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

void GenericTextureT::CopyFromAsync(const GPUQueue& queue,
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

GenericTextureView GenericTextureT::View(TextureReadMode mode) const
{
    return Impl()->View(mode);
}

bool GenericTextureT::HasRWView() const
{
    return Impl()->HasRWView();
}

SurfRefVariant GenericTextureT::RWView(uint32_t mipLevel)
{
    return Impl()->RWView(mipLevel);
}

const GPUDevice& GenericTextureT::Device() const
{
    return Impl()->Device();
}

uint32_t GenericTextureT::ChannelCount() const
{
    return Impl()->ChannelCount();
}


typename GenericTextureT::MipIsLoadedBits
GenericTextureT::ValidMips() const
{
    return isMipLoaded;
}

MRayColorSpaceEnum GenericTextureT::ColorSpace() const
{
    return colorSpace;
}

Float GenericTextureT::Gamma() const
{
    return gamma;
}

AttributeIsColor GenericTextureT::IsColor() const
{
    return isColor;
}

MRayPixelTypeRT GenericTextureT::PixelType() const
{
    return pixelType;
}

void GenericTextureT::SetAllMipsToLoaded()
{
    isMipLoaded.Set();
}

void GenericTextureT::SetColorSpace(MRayColorSpaceEnum e, Float gammaIn)
{
    colorSpace = e;
    gamma = gammaIn;
}