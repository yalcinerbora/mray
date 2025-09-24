#pragma once

#include "Core/TracerConstants.h"

#include "Device/GPUSystemForward.h"
#include "Device/GPUTypes.h"

#include "TextureView.h"
#include "TextureCommon.h"

class GenericTextureI;
struct TracerSurfRef;

// Generic Texture type
class alignas(8u) GenericTexture
{
    public:
    static constexpr size_t BuffSize        = 96u;
    static constexpr size_t BuffAlignment   = 8u;

    private:
    using MipIsLoadedBits = Bitset<TracerConstants::MaxTextureMipCount>;
    private:
    std::array<Byte, BuffSize> storage;
    // TODO: Could not cast the storage to the interface,
    //
    GenericTextureI*    impl;
    MRayPixelTypeRT     pixelType;
    MRayColorSpaceEnum  colorSpace;
    Float               gamma;
    AttributeIsColor    isColor;
    MipIsLoadedBits     isMipLoaded;

    GenericTextureI*         Impl();
    const GenericTextureI*   Impl() const;

    public:
    template<class T, class... Args>
                    GenericTexture(std::in_place_type_t<T>,
                                   MRayColorSpaceEnum, Float,
                                   AttributeIsColor, MRayPixelTypeRT,
                                   Args&&...);
    // TODO: Enable these later
                    GenericTexture(const GenericTexture&) = delete;
                    GenericTexture(GenericTexture&&) = delete;
    GenericTexture& operator=(const GenericTexture&) = delete;
    GenericTexture& operator=(GenericTexture&&) = delete;
                    ~GenericTexture();

    void            CommitMemory(const GPUQueue& queue,
                                 const TextureBackingMemory& deviceMem,
                                 size_t offset);
    size_t          Size() const;
    size_t          Alignment() const;
    uint32_t        MipCount() const;
    uint32_t        ChannelCount() const;
    MipIsLoadedBits ValidMips() const;

    //
    TextureExtent<3>    Extents() const;
    uint32_t            DimensionCount() const;
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      TransientData regionFrom);
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      Span<const Byte> regionFrom);
    void                CopyToAsync(Span<Byte> regionTo,
                                    const GPUQueue& queue,
                                    uint32_t mipLevel,
                                    const TextureExtent<3>& offset,
                                    const TextureExtent<3>& size) const;
    //
    GenericTextureView  View(TextureReadMode mode) const;
    bool                HasRWView() const;
    TracerSurfRef       RWView(uint32_t mipLevel);
    bool                IsBlockCompressed() const;

    const GPUDevice&    Device() const;

    // These are extra functionality that will be built on top of
    MRayColorSpaceEnum  ColorSpace() const;
    Float               Gamma() const;
    AttributeIsColor    IsColor() const;
    MRayPixelTypeRT     PixelType() const;

    void                SetAllMipsToLoaded();
    void                SetMipToLoaded(uint32_t mipLevel);
    void                SetColorSpace(MRayColorSpaceEnum, Float = Float(1));
};

// In order to prevent type leak we hand set these values
// It is statically checked on the cpp file.
using TextureMap = Map<TextureId, GenericTexture>;