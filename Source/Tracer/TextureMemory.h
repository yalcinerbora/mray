#pragma once

#include <bitset>

#include "Core/TracerI.h"

#include "Device/GPUSystemForward.h"
#include "Device/GPUTypes.h"

#include "TransientPool/TransientPool.h"

#include "GenericGroup.h"

//template<Texture T>
//using Texture2D = Texture<2, T>;

// Lets use type erasure on the host side
// unlike texture views, there are too many texture types
class CommonTextureI
{
    public:
    virtual ~CommonTextureI() = default;

    // These are fine, no special types here
    virtual void        CommitMemory(const GPUQueue& queue,
                                     const TextureBackingMemory& deviceMem,
                                     size_t offset) = 0;

    virtual size_t              Size() const = 0;
    virtual size_t              Alignment() const = 0;
    virtual uint32_t            MipCount() const = 0;
    virtual const GPUDevice&    Device() const = 0;

    // We can get away with dimension 3, and put INT32_MAX
    // to unused dimensions here
    virtual TextureExtent<3>    Extents() const = 0;
    // So we need this to determine the actual dimension
    virtual uint32_t            DimensionCount() const = 0;


    // Here is the hard part, incoming data must be the same type.
    // Thankfully, we type ereased the incoming data as "TransientData"
    // so we can utilize it. Again extents are maximum (3).
    //
    // Internals of this function will be a mess though.
    virtual void    CopyFromAsync(const GPUQueue& queue,
                                  uint32_t mipLevel,
                                  const TextureExtent<3>& offset,
                                  const TextureExtent<3>& size,
                                  TransientData regionFrom) = 0;

    // Another hard part, We can close the types here.
    // Only float convertible views are supported
    // TODO: This may be a limitation for rare cases maybe later?
    virtual GenericTextureView  View() const = 0;

    // And All Done!
};

// Generic Texture type
template<size_t StorageSize, size_t Alignment>
class alignas(Alignment) CommonTextureT
{
    using MipIsLoadedBits = std::bitset<TracerConstants::MaxTextureMipCount>;
    private:
    std::array<Byte, StorageSize> storage;
    // TODO: Could not cast the storate to the interface,
    //
    CommonTextureI*     impl;
    MRayPixelTypeRT     pixelType;
    MRayColorSpaceEnum  colorSpace;
    AttributeIsColor    isColor;
    MipIsLoadedBits     isMipLoaded;

    CommonTextureI*         Impl();
    const CommonTextureI*   Impl() const;

    public:
    template<class T, class... Args>
                    CommonTextureT(std::in_place_type_t<T>,
                                   MRayColorSpaceEnum, AttributeIsColor,
                                   MRayPixelTypeRT,
                                   Args&&...);
    // TODO: Enable these later
                    CommonTextureT(const CommonTextureT&) = delete;
                    CommonTextureT(CommonTextureT&&) = delete;
    CommonTextureT& operator=(const CommonTextureT&) = delete;
    CommonTextureT& operator=(CommonTextureT&&) = delete;
                    ~CommonTextureT();

    void            CommitMemory(const GPUQueue& queue,
                                 const TextureBackingMemory& deviceMem,
                                 size_t offset);
    size_t          Size() const;
    size_t          Alignment() const;
    uint32_t        MipCount() const;
    //
    TextureExtent<3>    Extents() const;
    uint32_t            DimensionCount() const;
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      TransientData regionFrom);
    GenericTextureView  View() const;
    const GPUDevice&    Device() const;

    // These are extra functionality that will be built on top of
    MRayColorSpaceEnum  ColorSpace() const;
    AttributeIsColor    IsColor() const;
    MRayPixelTypeRT     PixelType() const;

};

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
    const GPUDevice&    Device() const override;
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

using CommonTexture = CommonTextureT
<
    std::max(sizeof(TexDetail::Concept<Texture<3, Vector4>>),
             sizeof(TexDetail::Concept<Texture<2, Vector4>>)),
    std::max(alignof(TexDetail::Concept<Texture<3, Vector4>>),
             alignof(TexDetail::Concept<Texture<2, Vector4>>))
>;

class TextureMemory
{
    using GPUIterator       = GPUQueueIteratorRoundRobin;
    using TextureMap        = ThreadSafeMap<TextureId, CommonTexture>;
    using TextureMemList    = std::vector<TextureBackingMemory>;
    using TSTextureViewMap  = ThreadSafeMap<TextureId, GenericTextureView>;



    private:
    bool                    isCommitted = false;
    const GPUSystem&        gpuSystem;
    TextureMemList          texMemList;
    std::atomic_uint32_t    texCounter;
    TextureMap              textures;
    uint32_t                clampResolution;
    //
    std::atomic_uint32_t    gpuIndexCounter = 0;

    // The view map
    TSTextureViewMap        textureViews;

    template<uint32_t D>
    TextureId CreateTexture(const Vector<D, uint32_t>& size, uint32_t mipCount,
                            const MRayTextureParameters&);

    public:
    // Constructors & Destructor
                    TextureMemory(const GPUSystem&,
                                  uint32_t clampResolution = std::numeric_limits<uint32_t>::max());

    //
    TextureId       CreateTexture2D(const Vector2ui& size, uint32_t mipCount,
                                    const MRayTextureParameters&);
    TextureId       CreateTexture3D(const Vector3ui& size, uint32_t mipCount,
                                    const MRayTextureParameters&);

    void            CommitTextures();
    void            PushTextureData(TextureId, uint32_t mipLevel,
                                    TransientData data);

    MRayPixelTypeRT         GetPixelType(TextureId) const;
    const TextureViewMap&   TextureViews() const;
    void                    Clear();
    size_t                  GPUMemoryUsage() const;
};