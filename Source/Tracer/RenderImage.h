#pragma once

#include "Core/Types.h"
#include "Common/RenderImageStructs.h"
#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"
#include "Device/GPUAtomic.h"

struct RenderImageParams;
class RenderImage;


template <int32_t C>
class ImageSpan
{
    private:
    Span<Vector<C, Float>> dPixels;
    Span<Float> dWeights;
    Vector2i    extent;
    Vector2i    rangeStart;
    Vector2i    rangeEnd;

    public:
    MRAY_HOST   ImageSpan(const Span<Vector<C, Float>>& dPixelsIn,
                          const Span<Float>& dWeights,
                          const Vector2i& extent,
                          const Vector2i& start,
                          const Vector2i& end);
    //
    MRAY_HYBRID Vector2i        Extent() const;
    MRAY_HYBRID Vector2i        Start() const;
    MRAY_HYBRID Vector2i        End() const;
    MRAY_HYBRID uint32_t        LinearIndexFrom2D(const Vector2i& xy) const;
    MRAY_HYBRID Vector2i        LinearIndexTo2D(uint32_t pixelIndex) const;
    // Sample Related
    MRAY_GPU Float              FetchWeight(const Vector2i& xy) const;
    MRAY_GPU void               StoreWeight(Float val, const Vector2i& xy) const;
    MRAY_GPU Float              AddToWeightAtomic(Float val, const Vector2i& xy) const;
    // Pixel Related
    MRAY_GPU Vector<C, Float>   FetchPixel(const Vector2i& xy) const;
    MRAY_GPU void               StorePixel(const Vector<C, Float>& val,
                                               const Vector2i& xy) const;
    MRAY_GPU Vector<C, Float>   AddToPixelAtomic(const Vector<C, Float>& val,
                                                 const Vector2i& xy) const;
};

class ImageTiler
{
    public:
    using Range2D = std::array<Vector2ui, 2>;

    static Vector2ui   FindOptimumTileSize(const Vector2ui& imageSize,
                                           uint32_t parallelizationHint);

    private:
    RenderImage* renderBuffer = nullptr;

    // Full image resolution, This single image may be generated
    // by multiple tracers.
    Vector2ui   fullResolution  = Vector2ui::Zero();
    // This is the Image tilers responsible range
    // This range will be tiled
    Range2D     range           = {Vector2ui::Zero(), Vector2ui::Zero()};
    // Current tile count of the range defined above
    Vector2ui   tileCount           = Vector2ui::Zero();
    // Conservative tile size of a single tile
    Vector2ui   coveringTileSize    = Vector2ui::Zero();
    // Padded tile size (padding comes from the reconstruction filtering)
    // of the image
    Vector2ui   paddedTileSize      = Vector2ui::Zero();
    // This class is a state machine (iterator of some sort as well)
    // Initially the first tile is selected
    uint32_t    currentTile         = 0;
    Vector2ui   pixel1DRange        = Vector2ui::Zero();

    Vector2ui   ResponsibleSize() const;
    Vector2ui   GlobalTileStart() const;
    Vector2ui   GlobalTileEnd() const;
    public:
    // Constructors & Destructor
    ImageTiler() = default;
    ImageTiler(RenderImage* renderImage,
               const RenderImageParams& rIParams,
               uint32_t parallelizationHint,
               Vector2ui filterPadding = Vector2ui::Zero(),
               uint32_t channels = 3, uint32_t depth = 1);

    Vector2ui   FullResolution() const;
    Vector2ui   LocalTileStart() const;
    Vector2ui   LocalTileEnd() const;

    Vector2ui   Tile1DRange() const;
    Vector2ui   ConservativeTileSize() const;
    Vector2ui   CurrentTileSize() const;
    Vector2ui   CurrentTileIndex() const;
    Vector2ui   TileCount() const;
    void        NextTile();

    template<uint32_t C>
    ImageSpan<C>     GetTileSpan();

    Optional<RenderImageSection>
    TransferToHost(const GPUQueue& processQueue,
                   const GPUQueue& transferQueue);

};

class RenderImage
{
    private:
    const GPUSystem&    gpuSystem;
    uint32_t            importAlignment;
    // Mem related
    // According to the profiling this staging
    // transfer style was the most performant
    DeviceMemory        deviceMemory;
    Span<Float>         dPixels;
    Span<Float>         dWeights;
    // We use special "HostLocalAlignedMemory"
    // type here just because of Vulkan.
    HostLocalAlignedMemory  stagingMemory;
    Span<Float>             hPixels;
    Span<Float>             hWeights;
    //
    size_t              pixStartOffset      = 0;
    size_t              weightStartOffset   = 0;

    GPUSemaphoreView    sem;
    GPUFence            processCompleteFence;
    GPUFence            previousCopyCompleteFence;
    //
    uint32_t            channelCount    = 0;
    Vector2ui           extent          = Vector2ui::Zero();
    uint32_t            depth           = 0;

    public:
    // Constructors & Destructor
                        RenderImage(TimelineSemaphore* semaphore,
                                    uint32_t importAlignment,
                                    uint64_t initialSemCounter,
                                    const GPUSystem& gpuSystem);
                        RenderImage(const RenderImage&) = delete;
    RenderImage&        operator=(const RenderImage&) = delete;
                        ~RenderImage() = default;

    // Members
    // Access
    Span<Float>         Pixels();
    Span<Float>         Weights();
    //
    Vector2ui           Extents() const;
    uint32_t            Depth() const;
    uint32_t            ChannelCount() const;
    //
    void                ClearImage(const GPUQueue& queue);
    bool                Resize(const Vector2ui& extent,
                               uint32_t depth,
                               uint32_t channelCount);
    const GPUFence&     PrevCopyCompleteFence() const;

    // Synchronized Host access
    Optional<RenderImageSection>
                        TransferToHost(const GPUQueue& processQueue,
                                       const GPUQueue& copyQueue);

    Pair<const Byte*, size_t> SharedDataPtrAndSize() const;
};

template<uint32_t C>
ImageSpan<C> ImageTiler::GetTileSpan()
{
    // TODO: Change this
    Span<Byte> dPixels = Span<Byte>(reinterpret_cast<Byte*>(renderBuffer->Pixels().data()),
                                   renderBuffer->Pixels().size_bytes());

    auto dPixelsC = MemAlloc::RepurposeAlloc<Vector<C, Float>>(dPixels);
    return ImageSpan<C>(dPixelsC,
                        renderBuffer->Weights(),
                        Vector2i(CurrentTileSize()),
                        // TODO: Add padding
                        Vector2i::Zero(),
                        Vector2i(CurrentTileSize()));
}

template<int32_t C>
MRAY_HOST inline
ImageSpan<C>::ImageSpan(const Span<Vector<C, Float>>& dPixelsIn,
                        const Span<Float>& dWeightsIn,
                        const Vector2i& extentIn,
                        const Vector2i& startIn,
                        const Vector2i& endIn)
    : dPixels(dPixelsIn)
    , dWeights(dWeightsIn)
    , extent(extentIn)
    , rangeStart(startIn)
    , rangeEnd(endIn)
{}

template<int32_t C>
MRAY_HYBRID Vector2i ImageSpan<C>::Extent() const
{
    return extent;
}

template<int32_t C>
MRAY_HYBRID Vector2i ImageSpan<C>::Start() const
{
    return rangeStart;
}

template<int32_t C>
MRAY_HYBRID Vector2i ImageSpan<C>::End() const
{
    return rangeEnd;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
uint32_t ImageSpan<C>::LinearIndexFrom2D(const Vector2i& xy) const
{
    int32_t linear = xy[1] * extent[0] + xy[0];
    return static_cast<uint32_t>(linear);
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector2i ImageSpan<C>::LinearIndexTo2D(uint32_t linearIndex) const
{
    Vector2i result(linearIndex % extent[0],
                    linearIndex / extent[0]);
    return result;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Float ImageSpan<C>::FetchWeight(const Vector2i& xy) const
{
    uint32_t i = LinearIndexFrom2D(xy);
    return dWeights[i];
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
void ImageSpan<C>::StoreWeight(Float val, const Vector2i& xy) const
{
    uint32_t i = LinearIndexFrom2D(xy);
    dWeights[i] = val;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Float ImageSpan<C>::AddToWeightAtomic(Float val, const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return DeviceAtomic::AtomicAdd(dWeights[index], val);
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector<C, Float> ImageSpan<C>::FetchPixel(const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return dPixels[index];
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
void ImageSpan<C>::StorePixel(const Vector<C, Float>& val,
                                     const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    dPixels[index] = val;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector<C, Float> ImageSpan<C>::AddToPixelAtomic(const Vector<C, Float>& val,
                                                const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return DeviceAtomic::AtomicAdd(dPixels[index], val);
}

inline
Span<Float> RenderImage::Pixels()
{
    return dPixels;
}

inline
Span<Float> RenderImage::Weights()
{
    return dWeights;
}

inline
const GPUFence& RenderImage::PrevCopyCompleteFence() const
{
    return previousCopyCompleteFence;
}