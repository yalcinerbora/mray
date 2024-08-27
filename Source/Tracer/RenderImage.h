#pragma once

#include "Core/Types.h"
#include "Common/RenderImageStructs.h"
#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"
#include "Device/GPUAtomic.h"

struct RenderImageParams;
class RenderImage;

template <int32_t C>
class SubImageSpan
{
    private:
    Span<Vector<C, Float>>  dPixels;
    Span<Float>             dWeights;
    Vector2i                min;
    Vector2i                max;
    Vector2i                resolution;

    private:
    static MRAY_GPU Vector2i
    ChannelIndex(uint32_t channelLinear);

    public:
    MRAY_HOST   SubImageSpan(const Span<Vector<C, Float>>& dPixelsIn,
                             const Span<Float>& dWeights,
                             const Vector2ui& min,
                             const Vector2ui& max,
                             const Vector2ui& resolution);
    //
    MRAY_HYBRID Vector2i        Extent() const;
    MRAY_HYBRID Vector2i        Resolution() const;
    MRAY_HYBRID uint32_t        LinearIndexFrom2D(const Vector2i& xy) const;
    MRAY_HYBRID Vector2i        LinearIndexTo2D(uint32_t pixelIndex) const;
    // Sample Related
    MRAY_GPU Float              FetchWeight(const Vector2i& xy) const;
    MRAY_GPU void               StoreWeight(Float val, const Vector2i& xy) const;
    MRAY_GPU Float              AddToWeightAtomic(Float val, const Vector2i& xy) const;

    // Pixel Related
    MRAY_GPU Float              FetchPixelChannel(const Vector2i& xy,
                                                  uint32_t channelIndex) const;
    MRAY_GPU Vector<C, Float>   FetchPixelBulk(const Vector2i& xy) const;
    //
    MRAY_GPU void               StorePixelChannel(Float val,
                                                  const Vector2i& xy,
                                                  uint32_t channelIndex) const;
    MRAY_GPU void               StorePixelBulk(const Vector<C, Float>& val,
                                               const Vector2i& xy) const;
    //
    MRAY_GPU Float              AddToPixelChannelAtomic(Float val,
                                                        const Vector2i& xy,
                                                        uint32_t channelIndex) const;
    MRAY_GPU Vector<C, Float>   AddToPixelBulkAtomic(const Vector<C, Float>& val,
                                                     const Vector2i& xy) const;
};

class ImageTiler
{
    public:
    using Range2D = std::array<Vector2ui, 2>;

    static Vector2ui   FindOptimumTile(const Vector2ui& imageSize,
                                       uint32_t parallelizationHint);

    private:
    RenderImage* renderBuffer = nullptr;

    // Full image resolution, This single image may be generated
    // by multiple tracers.
    Vector2ui   fullResolution  = Vector2ui::Zero();
    Range2D     imageRange      = {Vector2ui::Zero(), Vector2ui::Zero()};
    // Current tile count of the sub range defined above
    Vector2ui   tileCount           = Vector2ui::Zero();
    Vector2ui   coveringTileSize    = Vector2ui::Zero();
    Vector2ui   renderBufferSize    = Vector2ui::Zero();
    uint32_t    currentTile         = 0;

    Vector2ui ResponsibleSize() const;

    public:
    // Constructors & Destructor
    ImageTiler() = default;
    ImageTiler(RenderImage* renderImage,
               const RenderImageParams& rIParams,
               uint32_t parallelizationHint,
               Vector2ui filterPadding = Vector2ui::Zero(),
               uint32_t channels = 3, uint32_t depth = 1);

    Vector2ui   CurrentTileSize();
    Vector2ui   CurrentTileIndex();
    Vector2ui   TileCount();
    void        NextTile();

    template<uint32_t C>
    SubImageSpan<C>     AsSubspan();

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
    Span<Float>         dSamples;
    // We use special "HostLocalAlignedMemory"
    // type here just because of Vulkan.
    HostLocalAlignedMemory  stagingMemory;
    Span<Float>             hPixels;
    Span<Float>             hSamples;
    //
    size_t              pixStartOffset      = 0;
    size_t              sampleStartOffset   = 0;

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
    Span<Float>         Samples();
    //
    Vector2ui           Extents() const;
    uint32_t            Depth() const;
    uint32_t            ChannelCount() const;
    //
    void                ClearImage(const GPUQueue& queue);
    RenderBufferInfo    GetBufferInfo(MRayColorSpaceEnum colorspace,
                                      const Vector2ui& resolution,
                                      uint32_t depth);
    bool                Resize(const Vector2ui& extent,
                               uint32_t depth,
                               uint32_t channelCount);
    const GPUFence&     PrevCopyCompleteFence() const;

    // Synchronized Host access
    Optional<RenderImageSection>
                        TransferToHost(const GPUQueue& processQueue,
                                       const GPUQueue& copyQueue);
};

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector2i SubImageSpan<C>::ChannelIndex(uint32_t channelLinear)
{
    int32_t channel = static_cast<int32_t>(channelLinear);
    return Vector2i(channel % C, channel / C);
}

template<int32_t C>
MRAY_HOST inline
SubImageSpan<C>::SubImageSpan(const Span<Vector<C, Float>>& dPixelsIn,
                              const Span<Float>& dWeightsIn,
                              const Vector2ui& minIn,
                              const Vector2ui& maxIn,
                              const Vector2ui& resolutionIn)
    : dPixels(dPixelsIn)
    , dWeights(dWeightsIn)
    , min(minIn[0], minIn[1])
    , max(maxIn[0], maxIn[1])
    , resolution(resolutionIn[0], resolutionIn[1])
{}

template<int32_t C>
MRAY_HYBRID Vector2i SubImageSpan<C>::Extent() const
{
    return max - min;
}

template<int32_t C>
MRAY_HYBRID Vector2i SubImageSpan<C>::Resolution() const
{
    return resolution;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
uint32_t SubImageSpan<C>::LinearIndexFrom2D(const Vector2i& xy) const
{
    int32_t linear = xy[1] * resolution[0] + xy[0];
    return static_cast<uint32_t>(linear);
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector2i SubImageSpan<C>::LinearIndexTo2D(uint32_t linearIndex) const
{
    Vector2i result(linearIndex % resolution[0],
                    linearIndex / resolution[0]);
    return result;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Float SubImageSpan<C>::FetchWeight(const Vector2i& xy) const
{
    uint32_t i = LinearIndexFrom2D(xy);
    return dWeights[i];
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
void SubImageSpan<C>::StoreWeight(Float val, const Vector2i& xy) const
{
    uint32_t i = LinearIndexFrom2D(xy);
    dWeights[i] = val;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Float SubImageSpan<C>::AddToWeightAtomic(Float val, const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return DeviceAtomic::AtomicAdd(dWeights[index], val);
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Float SubImageSpan<C>::FetchPixelChannel(const Vector2i& xy,
                                         uint32_t channelIndex) const
{
    Vector2i c = ChannelIndex(channelIndex);
    uint32_t index = static_cast<int32_t>(LinearIndexFrom2D(xy));
    return dPixels[index * c[1]][c[0]];
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector<C, Float> SubImageSpan<C>::FetchPixelBulk(const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return dPixels[index];
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
void SubImageSpan<C>::StorePixelChannel(Float val,
                                        const Vector2i& xy,
                                        uint32_t channelIndex) const
{
    Vector2i c = ChannelIndex(channelIndex);
    uint32_t index = static_cast<int32_t>(LinearIndexFrom2D(xy));
    dPixels[index * c[1]][c[0]] = val;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
void SubImageSpan<C>::StorePixelBulk(const Vector<C, Float>& val,
                                     const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    dPixels[index] = val;
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Float SubImageSpan<C>::AddToPixelChannelAtomic(Float val, const Vector2i& xy,
                                               uint32_t channelIndex) const
{
    Vector2i c = ChannelIndex(channelIndex);
    uint32_t index = static_cast<int32_t>(LinearIndexFrom2D(xy));
    DeviceAtomic::AtomicAdd(dPixels[index * c[1]][c[0]], val);
}

template<int32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector<C, Float> SubImageSpan<C>::AddToPixelBulkAtomic(const Vector<C, Float>& val,
                                                       const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    DeviceAtomic::AtomicAdd(dPixels[index], val);
}

inline
Span<Float> RenderImage::Pixels()
{
    return dPixels;
}

inline
Span<Float> RenderImage::Samples()
{
    return dSamples;
}

inline
const GPUFence& RenderImage::PrevCopyCompleteFence() const
{
    return previousCopyCompleteFence;
}

template<uint32_t C>
SubImageSpan<C> ImageTiler::AsSubspan()
{
    assert(C == renderBuffer->Depth());
    // Alias the buffer
    using PixelType = Span<Vector<C, Float>>;
    PixelType dPixelSpan;// = MemAlloc::RepurposeAlloc<PixelType>(dPixels);

    //return SubImageSpan<C>(dPixelSpan, dSamples, min, max, resolution);
}