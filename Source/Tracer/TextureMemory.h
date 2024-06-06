#pragma once

#include "Device/GPUSystemForward.h"
#include "Device/GPUTypes.h"

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
    virtual size_t      Size() const = 0;
    virtual size_t      Alignment() const = 0;
    virtual uint32_t    MipCount() const = 0;

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

template<size_t S, size_t A>
class CommonTextureT;

using CommonTexture = CommonTextureT<std::max(sizeof(Texture<3, Vector4>),
                                              sizeof(Texture<2, Vector4>)),
                                     std::max(alignof(Texture<3, Vector4>),
                                              alignof(Texture<2, Vector4>))>;

class TextureMemory
{
    private:
    const GPUSystem&                gpuSystem;
    std::atomic_uint32_t            texCounter;
    //Map<TextureId, CommonTexture>   textures;


    public:
    // Constructors & Destructor
                TextureMemory(const GPUSystem&);

    //
    TextureId   CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                MRayPixelTypeRT pixType,
                                AttributeIsColor);
};