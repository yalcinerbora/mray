#include "TexViewRenderer.h"
#include "TracerBase.h"

#include "Device/GPUSystem.hpp"

#include "Device/GPUAtomic.h"
class SubImageSpan
{
    private:
    Span<Float> dPixels;
    Span<Float> dSamples;

    Vector2ui min;
    Vector2ui max;
    Vector2ui resolution;

    template<uint32_t C>
    MRAY_GPU uint32_t           LinearIndex(const Vector2ui& xy) const;
    MRAY_GPU Float              FetchSample(const Vector2ui& xy) const;
    template<uint32_t C>
    MRAY_GPU Vector<C, Float>   FetchPixel(const Vector2ui& xy) const;
    template<uint32_t C>
    MRAY_GPU void               StorePixel(const Vector<C, Float>& val,
                                           const Vector2ui& xy);
    template<uint32_t C>
    MRAY_GPU Vector<C, Float>   AddToPixelAtomic(const Vector<C, Float>& val,
                                                 const Vector2ui& xy);
    MRAY_GPU Float              AddToSampleAtomic(Float val, const Vector2ui& xy);
};

template<uint32_t C>
MRAY_GPU MRAY_GPU_INLINE
uint32_t SubImageSpan::LinearIndex(const Vector2ui& xy) const
{
    uint32_t yGlobal = (min[1] + xy[1]);
    uint32_t xGlobal = (min[0] + xy[0]);
    uint32_t linear = (yGlobal * resolution[1] + xGlobal) * C;
    return linear;
}

MRAY_GPU MRAY_GPU_INLINE
Float SubImageSpan::FetchSample(const Vector2ui& xy) const
{
    uint32_t i = LinearIndex<1>(xy);
    return dSamples[i];
}

template<uint32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector<C, Float> SubImageSpan::FetchPixel(const Vector2ui& xy) const
{
    uint32_t index = LinearIndex<C>(xy);
    Vector<C, Float> r;
    UNROLL_LOOP
    for(uint32_t i = 0; i < C; i++)
        r[i] = dPixels[index + i];
    return r;
}

template<uint32_t C>
MRAY_GPU MRAY_GPU_INLINE
void SubImageSpan::StorePixel(const Vector<C, Float>& val,
                              const Vector2ui& xy)
{
    uint32_t index = LinearIndex<C>(xy);
    UNROLL_LOOP
    for(uint32_t i = 0; i < C; i++)
        dPixels[index + i] = val[i];
}

template<uint32_t C>
MRAY_GPU MRAY_GPU_INLINE
Vector<C, Float> SubImageSpan::AddToPixelAtomic(const Vector<C, Float>& val,
                                                const Vector2ui& xy)
{
    Vector<C, Float> result;
    uint32_t index = LinearIndex<C>(xy);
    UNROLL_LOOP
    for(uint32_t i = 0; i < C; i++)
    {
        result[i] = DeviceAtomic::AtomicAdd(dPixels[index + i], val[i]);
    }
    return result;
}

MRAY_GPU MRAY_GPU_INLINE
Float SubImageSpan::AddToSampleAtomic(Float val, const Vector2ui& xy)
{
    uint32_t index = LinearIndex<0>(xy);
    return DeviceAtomic::AtomicAdd(dPixels[index], val);
}


TexViewRenderer::TexViewRenderer(RenderImagePtr rb, TracerView tv,
                             const GPUSystem& s)
    : RendererT(rb, tv, s)
{}

MRayError TexViewRenderer::Commit()
{
    if(rendering)
    currentOptions = newOptions;
    return MRayError::OK;
}

typename TexViewRenderer::AttribInfoList
TexViewRenderer::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP", MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY}
    };
}

RendererOptionPack TexViewRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));
    return result;
}

void TexViewRenderer::PushAttribute(uint32_t attributeIndex,
                                    TransientData data, const GPUQueue&)
{
    if(attributeIndex != 0)
        throw MRayError("{} Unkown attribute index {}",
                        TypeName(), attributeIndex);
    newOptions.totalSPP = data.AccessAs<uint32_t>()[0];
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t FilterRadiusToPixelWH(Float filterRadius)
{
    // At every 0.5 increment conservative pixel estimate is increasing
    // [0]          = Single Pixel (Special Case)
    // (0, 0.5]     = 2x2
    // (0.5, 1]     = 3x3
    // (1, 1.5]     = 4x4
    // (1.5, 2]     = 5x5
    // etc...
    if(filterRadius == Float(0)) return 1;
    // Do division
    uint32_t quot = static_cast<uint32_t>(filterRadius / Float(0.5));
    float remainder = std::fmod(filterRadius, Float(0.5));
    // Exact divisions reside on previous segment
    if(remainder == Float(0)) quot -= 1;
    uint32_t result = quot + 2;
    return result;
}

uint32_t FindOptimumTile(uint32_t regionSize,
                         uint32_t tileSize)
{
    // Find optimal tile size that evenly divides the image
    // This may not happen (width or height is prime)
    // then expand the tile size to pass the edge barely.
    if(regionSize < tileSize) return regionSize;

    // Divide and find a tileCount
    uint32_t tCount = MathFunctions::DivideUp(regionSize, tileSize);
    uint32_t result = regionSize / tCount;
    uint32_t residual = regionSize % tCount;
    // All file no pixel is left.
    if(residual == 0) return result;

    // Not evenly divisible now expand the tile
    residual = MathFunctions::DivideUp(residual, tCount);
    result += residual;
    return result;
}

RenderBufferInfo TexViewRenderer::StartRender(const RenderImageParams& params,
                                              const CameraKey&)
{
    // Calculate tile size according to the parallelization hint
    uint32_t parallelHint = 1 << 21;
    uint32_t tileHint = static_cast<int32_t>(std::round(std::sqrt(parallelHint)));
    Vector2ui imgRegion = params.regionMax - params.regionMin;
    // Add some tolerance (%30)
    tileHint = uint32_t(Float(0.3) * Float(tileHint));
    Vector2ui tileSize = Vector2ui(FindOptimumTile(imgRegion[0], tileHint),
                                   FindOptimumTile(imgRegion[1], tileHint));

    //Vector2ui extraPixels = FilterSize;


    // Tiled Render Buffer
    // Access tile

    using enum MRayColorSpaceEnum;
    renderBuffer = std::make_shared<RenderImage>(params, 1,
                                                 MR_ACES_CG,
                                                 gpuSystem);
    return renderBuffer->GetBufferInfo();
}

MRAY_KERNEL
void TestWrite(MRAY_GRID_CONSTANT const Span<Float> pixels,
               MRAY_GRID_CONSTANT const Span<Float> samples,
               MRAY_GRID_CONSTANT const uint32_t colorIndex)
{
    KernelCallParams kp;
    uint32_t pixelChannels = static_cast<uint32_t>(pixels.size());
    for(uint32_t i = kp.GlobalId(); i < pixelChannels; i += kp.TotalSize())
    {
        if(i < samples.size())
            samples[i] = 1.0;
        pixels[i] = (colorIndex % 2 == 0) ? 1.0f : 0.0f;
    }
}

RendererOutput TexViewRenderer::DoRender()
{
    using namespace std::string_view_literals;

    const GPUQueue& processQueue = gpuSystem.BestDevice().GetQueue(0);
    const GPUQueue& transferQueue = gpuSystem.BestDevice().GetQueue(QueuePerDevice - 1);
    Span<Float> pixels = renderBuffer->Pixels();
    Span<Float> samples = renderBuffer->Samples();
    uint32_t colorIndex = pixelIndex;
    processQueue.IssueSaturatingKernel<TestWrite>
    (
        "TexTest"sv,
        KernelIssueParams{.workCount = static_cast<uint32_t>(pixels.size())},
        //
        pixels,
        samples,
        colorIndex
    );

    RenderImageSection renderOut = renderBuffer->GetHostView(processQueue,
                                                             transferQueue);

    pixelIndex++;
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            3.2,
            "M paths/s",
            0.0,
            "spp",
            0.0,
            renderBuffer->Resolution(),
            MRayColorSpaceEnum::MR_ACES_CG
        },
        .imageOut = renderOut
    };
}

void TexViewRenderer::StopRender()
{}