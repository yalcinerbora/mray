#include "TextureMemory.h"
#include "SurfaceView.h"

GenericTextureI* GenericTexture::Impl()
{
    // TODO: Are these correct?
    GenericTextureI* ptr = reinterpret_cast<GenericTextureI*>(storage.data());
    return std::launder(ptr);
}

const GenericTextureI* GenericTexture::Impl() const
{
    // TODO: Are these correct?
    const GenericTextureI* ptr = reinterpret_cast<const GenericTextureI*>(storage.data());
    return std::launder(ptr);
}

GenericTexture::~GenericTexture()
{
    std::destroy_at(Impl());
}

void GenericTexture::CommitMemory(const GPUQueue& queue,
                                  const TextureBackingMemory& deviceMem,
                                  size_t offset)
{
    Impl()->CommitMemory(queue, deviceMem, offset);
}

size_t GenericTexture::Size() const
{
    return Impl()->Size();
}

size_t GenericTexture::Alignment() const
{
    return Impl()->Alignment();
}

uint32_t GenericTexture::MipCount() const
{
    return Impl()->MipCount();
}

TextureExtent<3> GenericTexture::Extents() const
{
    return Impl()->Extents();
}

uint32_t GenericTexture::DimensionCount() const
{
    return Impl()->DimensionCount();
}

void GenericTexture::CopyFromAsync(const GPUQueue& queue,
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

void GenericTexture::CopyFromAsync(const GPUQueue& queue,
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

void GenericTexture::CopyToAsync(Span<Byte> regionTo,
                                 const GPUQueue& queue,
                                 uint32_t mipLevel,
                                 const TextureExtent<3>& offset,
                                 const TextureExtent<3>& size) const
{
    Impl()->CopyToAsync(regionTo, queue, mipLevel,
                        offset, size);
}

GenericTextureView GenericTexture::View(TextureReadMode mode) const
{
    return Impl()->View(mode);
}

bool GenericTexture::HasRWView() const
{
    return Impl()->HasRWView();
}

bool GenericTexture::IsBlockCompressed() const
{
    return Impl()->IsBlockCompressed();
}

TracerSurfRef GenericTexture::RWView(uint32_t mipLevel)
{
    return Impl()->RWView(mipLevel);
}

const GPUDevice& GenericTexture::Device() const
{
    return Impl()->Device();
}

uint32_t GenericTexture::ChannelCount() const
{
    return Impl()->ChannelCount();
}


typename GenericTexture::MipIsLoadedBits
GenericTexture::ValidMips() const
{
    return isMipLoaded;
}

MRayColorSpaceEnum GenericTexture::ColorSpace() const
{
    return colorSpace;
}

Float GenericTexture::Gamma() const
{
    return gamma;
}

AttributeIsColor GenericTexture::IsColor() const
{
    return isColor;
}

MRayPixelTypeRT GenericTexture::PixelType() const
{
    return pixelType;
}

void GenericTexture::SetAllMipsToLoaded()
{
    isMipLoaded.Set();
}

void GenericTexture::SetMipToLoaded(uint32_t mipLevel)
{
    isMipLoaded[mipLevel] = true;
}

void GenericTexture::SetColorSpace(MRayColorSpaceEnum e, Float gammaIn)
{
    colorSpace = e;
    gamma = gammaIn;
}