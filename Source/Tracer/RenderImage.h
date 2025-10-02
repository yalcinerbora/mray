#pragma once

#include "Core/Types.h"
#include "Common/RenderImageStructs.h"
#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"
#include "Device/GPUAtomic.h"

struct RenderImageParams;
class RenderImage;

class ImageSpan
{
    //private:
    public:
    Span<Float> dPixelsR;
    Span<Float> dPixelsG;
    Span<Float> dPixelsB;
    Span<Float> dWeights;
    //
    Vector2i    extent;
    Vector2i    rangeStart;
    Vector2i    rangeEnd;

    public:
    MRAY_HOST   ImageSpan(const Span<Float>& dPixelsR,
                          const Span<Float>& dPixelsG,
                          const Span<Float>& dPixelsB,
                          const Span<Float>& dWeights,
                          const Vector2i& extent,
                          const Vector2i& start,
                          const Vector2i& end);
    //
    MR_HF_DECL Vector2i Extent() const;
    MR_HF_DECL Vector2i Start() const;
    MR_HF_DECL Vector2i End() const;
    MR_HF_DECL uint32_t LinearIndexFrom2D(const Vector2i& xy) const;
    MR_HF_DECL Vector2i LinearIndexTo2D(uint32_t pixelIndex) const;
    // Sample Related
    MR_GF_DECL Float    FetchWeight(const Vector2i& xy) const;
    MR_GF_DECL void     StoreWeight(Float val, const Vector2i& xy) const;
    MR_GF_DECL Float    AddToWeightAtomic(Float val, const Vector2i& xy) const;
    // Pixel Related
    MR_GF_DECL Vector3  FetchPixel(const Vector2i& xy) const;
    MR_GF_DECL void     StorePixel(const Vector3& val,
                                   const Vector2i& xy) const;
    MR_GF_DECL Vector3  AddToPixelAtomic(const Vector3& val,
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

    Vector2ui   GlobalTileStart() const;
    Vector2ui   GlobalTileEnd() const;
    public:
    // Constructors & Destructor
    ImageTiler() = default;
    ImageTiler(RenderImage* renderImage,
               const RenderImageParams& rIParams,
               uint32_t parallelizationHint,
               Vector2ui filterPadding = Vector2ui::Zero());

    Vector2ui   FullResolution() const;
    Vector2ui   RegionSize() const;
    Vector2ui   LocalTileStart() const;
    Vector2ui   LocalTileEnd() const;

    Vector2ui   ConservativeTileSize() const;
    Vector2ui   CurrentTileSize() const;
    Vector2ui   CurrentTileIndex() const;
    uint32_t    CurrentTileIndex1D() const;
    Vector2ui   TileCount() const;
    void        NextTile();

    ImageSpan   GetTileSpan();

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
    Span<Float>         dPixelsAll;
    Span<Float>         dPixelsR;
    Span<Float>         dPixelsG;
    Span<Float>         dPixelsB;
    Span<Float>         dWeights;
    // We use special "HostLocalAlignedMemory"
    // type here just because of Vulkan.
    HostLocalAlignedMemory  stagingMemory;
    Span<Float>             hPixelsAll;
    Span<Float>             hPixelsR;
    Span<Float>             hPixelsG;
    Span<Float>             hPixelsB;
    Span<Float>             hWeights;
    //
    std::array<size_t, 3>   pixStartOffsets     = {};
    size_t                  weightStartOffset   = 0;

    GPUSemaphoreView    sem;
    GPUFence            processCompleteFence;
    GPUFence            previousCopyCompleteFence;
    //
    Vector2ui           extent          = Vector2ui::Zero();

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
    std::array<Span<Float>, 3>  Pixels();
    Span<Float>                 Weights();
    //
    Vector2ui           Extents() const;
    uint32_t            ChannelCount() const;
    //
    void                ClearImage(const GPUQueue& queue);
    bool                Resize(const Vector2ui& extent);
    const GPUFence&     PrevCopyCompleteFence() const;

    // Synchronized Host access
    Optional<RenderImageSection>
                        TransferToHost(const GPUQueue& processQueue,
                                       const GPUQueue& copyQueue);

    Pair<const Byte*, size_t> SharedDataPtrAndSize() const;
};

MRAY_HOST inline
ImageSpan::ImageSpan(const Span<Float>& dPixelsRIn,
                     const Span<Float>& dPixelsGIn,
                     const Span<Float>& dPixelsBIn,
                     const Span<Float>& dWeightsIn,
                     const Vector2i& extentIn,
                     const Vector2i& startIn,
                     const Vector2i& endIn)
    : dPixelsR(dPixelsRIn)
    , dPixelsG(dPixelsGIn)
    , dPixelsB(dPixelsBIn)
    , dWeights(dWeightsIn)
    , extent(extentIn)
    , rangeStart(startIn)
    , rangeEnd(endIn)
{}

MR_HF_DEF
Vector2i ImageSpan::Extent() const
{
    return extent;
}

MR_HF_DEF
Vector2i ImageSpan::Start() const
{
    return rangeStart;
}

MR_HF_DEF
Vector2i ImageSpan::End() const
{
    return rangeEnd;
}

MR_HF_DEF
uint32_t ImageSpan::LinearIndexFrom2D(const Vector2i& xy) const
{
    int32_t linear = xy[1] * extent[0] + xy[0];
    return static_cast<uint32_t>(linear);
}

MR_HF_DEF
Vector2i ImageSpan::LinearIndexTo2D(uint32_t linearIndex) const
{
    Vector2i result(linearIndex % uint32_t(extent[0]),
                    linearIndex / uint32_t(extent[0]));
    return result;
}

MR_GF_DEF
Float ImageSpan::FetchWeight(const Vector2i& xy) const
{
    uint32_t i = LinearIndexFrom2D(xy);
    return dWeights[i];
}

MR_GF_DEF
void ImageSpan::StoreWeight(Float val, const Vector2i& xy) const
{
    uint32_t i = LinearIndexFrom2D(xy);
    dWeights[i] = val;
}

MR_GF_DEF
Float ImageSpan::AddToWeightAtomic(Float val, const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return DeviceAtomic::AtomicAdd(dWeights[index], val);
}

MR_GF_DEF
Vector3 ImageSpan::FetchPixel(const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return Vector3(dPixelsR[index], dPixelsG[index], dPixelsB[index]);
}

MR_GF_DEF
void ImageSpan::StorePixel(const Vector3& val,
                           const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    dPixelsR[index] = val[0];
    dPixelsG[index] = val[1];
    dPixelsB[index] = val[2];
}

MR_GF_DEF
Vector3 ImageSpan::AddToPixelAtomic(const Vector3& val,
                                    const Vector2i& xy) const
{
    uint32_t index = LinearIndexFrom2D(xy);
    return Vector3(DeviceAtomic::AtomicAdd(dPixelsR[index], val[0]),
                   DeviceAtomic::AtomicAdd(dPixelsG[index], val[1]),
                   DeviceAtomic::AtomicAdd(dPixelsB[index], val[2]));
}

inline
ImageSpan ImageTiler::GetTileSpan()
{
    std::array<Span<Float>, 3> pixels = renderBuffer->Pixels();

    return ImageSpan(pixels[0],
                     pixels[1],
                     pixels[2],
                     renderBuffer->Weights(),
                     Vector2i(CurrentTileSize()),
                     // TODO: Add padding
                     Vector2i::Zero(),
                     Vector2i(CurrentTileSize()));
}


inline
std::array<Span<Float>, 3> RenderImage::Pixels()
{
    return {dPixelsR, dPixelsG, dPixelsB};
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