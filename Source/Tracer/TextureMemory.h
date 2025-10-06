#pragma once

#include "Core/TracerI.h"

#include "Device/GPUSystemForward.h"
#include "Device/GPUTypes.h"
#include "Device/GPUTexture.h"

#include "TransientPool/TransientPool.h"

#include "TextureFilter.h"
#include "Texture.h"

struct TracerSurfRef;
struct TracerSurfView;

struct TexClampParameters
{
    Vector2ui       inputMaxRes;
    uint16_t        filteredMipLevel    = 0;
    uint16_t        ignoredMipCount     = 0;
    bool            willBeFiltered      = false;
    // TODO: Due to heavy template cost of SurfRefVariant
    // (and it is parsed by most of the cu files)
    // we heap allocate the thing.
    std::unique_ptr<TracerSurfRef> surface;

    // Constructors & Destructor
    TexClampParameters() = default;
    TexClampParameters(TexClampParameters&&) = default;
    // To accept fully incomplete type, define the destructor
    // (we will default it outside)
    ~TexClampParameters();
};

// Lets use type erasure on the host side
// unlike texture views, there are too many texture types
class GenericTextureI
{
    public:
    virtual ~GenericTextureI() = default;

    // These are fine, no special types here
    virtual void        CommitMemory(const GPUQueue& queue,
                                     const TextureBackingMemory& deviceMem,
                                     size_t offset) = 0;

    virtual size_t              Size() const = 0;
    virtual size_t              Alignment() const = 0;
    virtual uint32_t            MipCount() const = 0;
    virtual uint32_t            ChannelCount() const = 0;
    virtual const GPUDevice&    Device() const = 0;

    // We can get away with dimension 3, and put INT32_MAX
    // to unused dimensions here
    virtual TextureExtent<3>    Extents() const = 0;
    // So we need this to determine the actual dimension
    virtual uint32_t            DimensionCount() const = 0;


    // Here is the hard part, incoming data must be the same type.
    // Thankfully, we type erased the incoming data as "TransientData"
    // so we can utilize it. Again extents are maximum (3).
    //
    // Internals of this function will be a mess though.
    virtual void    CopyFromAsync(const GPUQueue& queue,
                                  uint32_t mipLevel,
                                  const TextureExtent<3>& offset,
                                  const TextureExtent<3>& size,
                                  TransientData regionFrom) = 0;

    // "Any" buffer version, this will be used to copy data from device
    // memory
    virtual void    CopyFromAsync(const GPUQueue& queue,
                                  uint32_t mipLevel,
                                  const TextureExtent<3>& offset,
                                  const TextureExtent<3>& size,
                                  Span<const Byte> regionFrom) = 0;
    // Opposite version, only for BC textures atm
    virtual void    CopyToAsync(Span<Byte> regionTo,
                                const GPUQueue& queue,
                                uint32_t mipLevel,
                                const TextureExtent<3>& offset,
                                const TextureExtent<3>& size) const = 0;

    // Another hard part, We can close the types here.
    // Only float convertible views are supported
    // TODO: This may be a limitation for rare cases maybe later?
    virtual GenericTextureView  View(TextureReadMode mode) const = 0;

    // Writable view only used for mipmap generation,
    // Only 2D and not block-compressed textures are supported
    virtual bool            HasRWView() const = 0;
    virtual TracerSurfRef   RWView(uint32_t mipLevel) = 0;
    virtual bool            IsBlockCompressed() const = 0;

    // And All Done!
};

class TextureMemory
{
    using GPUIterator       = GPUQueueIteratorRoundRobin;
    using TextureMemList    = std::vector<TextureBackingMemory>;
    using TSClampMap        = ThreadSafeMap<TextureId, TexClampParameters>;
    using TSTextureMap      = ThreadSafeMap<TextureId, GenericTexture>;
    using TSTextureViewMap  = ThreadSafeMap<TextureId, GenericTextureView>;
    using TextureFilterPtr = std::unique_ptr<TextureFilterI>;


    private:
    const GPUSystem&            gpuSystem;
    const TracerParameters&     tracerParams;
    TextureFilterPtr            mipGenFilter;
    // State
    TextureMemList          texMemList;
    std::atomic_uint32_t    texCounter;
    TSTextureMap            textures;
    // Texture clamp related
    TSClampMap              texClampParams;
    //
    std::atomic_uint32_t    gpuIndexCounter = 0;
    // The view map
    TSTextureViewMap        textureViews;
    // Sizeof Temporary filterbuffer
    size_t                  texClampBufferSize = 0;
    DeviceLocalMemory       texClampBuffer;
    std::mutex              texClampMutex;

    // Impl functions
    void            GenerateMipmaps();
    void            ConvertColorspaces();
    template<uint32_t D>
    TextureId       CreateTexture(const Vector<D, uint32_t>& size,
                                  uint32_t mipCount,
                                  const MRayTextureParameters&);

    public:
    // Constructors & Destructor
                    TextureMemory(const GPUSystem&,
                                  const TracerParameters&,
                                  const FilterGeneratorMap&);
    //
    TextureId       CreateTexture2D(const Vector2ui& size, uint32_t mipCount,
                                    const MRayTextureParameters&);
    TextureId       CreateTexture3D(const Vector3ui& size, uint32_t mipCount,
                                    const MRayTextureParameters&);

    void            CommitTextures();
    void            PushTextureData(TextureId, uint32_t mipLevel,
                                    TransientData data);
    void            Finalize();

    const TextureViewMap&   TextureViews() const;
    const TextureMap&       Textures() const;

    MRayPixelTypeRT         GetPixelType(TextureId) const;
    void                    Clear();
    size_t                  GPUMemoryUsage() const;
};