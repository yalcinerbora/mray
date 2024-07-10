#include "TexViewRenderer.h"
#include "TracerBase.h"
#include "ColorFunctions.h"
#include "CommonTexture.hpp"

#include "Device/GPUSystem.hpp"

#include "Core/Timer.h"


MRAY_KERNEL
void KCColorTiles(MRAY_GRID_CONSTANT const Span<Float> pixels,
                  MRAY_GRID_CONSTANT const Span<Float> samples,
                  MRAY_GRID_CONSTANT const uint32_t colorIndex,
                  MRAY_GRID_CONSTANT const Vector2ui resolution,
                  MRAY_GRID_CONSTANT const uint32_t channelCount)
{
    KernelCallParams kp;

    uint32_t totalPix = resolution.Multiply();
    assert(totalPix * channelCount <= pixels.size());
    assert(totalPix * channelCount <= samples.size());

    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        uint32_t pixIndex = i * channelCount;
        Vector3 color = Color::RandomColorRGB(colorIndex);

        samples[pixIndex] = Float(1);
        pixels[pixIndex + 0] = color[0];
        pixels[pixIndex + 1] = color[1];
        pixels[pixIndex + 2] = color[2];
    }
}

TexViewRenderer::TexViewRenderer(const RenderImagePtr& rb,
                                 TracerView tv, const GPUSystem& s)
    : RendererT(rb, tv, s)
{
    // Pre-generate list
    textures.clear();
    textures.reserve(tracerView.textures.size());
    for(const auto& tex : tracerView.textures)
    {
        // Skip 1D/3D textures we can not render those
        if(tex.second.DimensionCount() != 2) continue;

        textures.push_back(&tex.second);
    }
    if(textures.empty())
    {
        MRAY_WARNING_LOG("[(R)TexView] No textures are present "
                         "in the tracer. Rendering nothing!");
    }
}

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

RenderBufferInfo TexViewRenderer::StartRender(const RenderImageParams&,
                                              const CameraKey&,
                                              uint32_t customLogicIndex0,
                                              uint32_t customLogicIndex1)
{
    // Skip if nothing to show
    if(textures.empty()) return RenderBufferInfo
    {
        .data = nullptr,
        .totalSize = 0,
        .renderColorSpace = tracerView.tracerParams.globalTextureColorSpace,
        .resolution = Vector2ui::Zero(),
        .depth = 0
    };

    // Find the texture index
    using MathFunctions::Roll;
    textureIndex = Roll(int32_t(customLogicIndex0), 0,
                        int32_t(textures.size()));
    const CommonTexture* t = textures[textureIndex];
    mipIndex = Roll(int32_t(customLogicIndex1), 0,
                    int32_t(t->MipCount()));

    // Calculate tile size according to the parallelization hint
    uint32_t parallelHint = tracerView.tracerParams.parallelizationHint;
    uint32_t tileHint = static_cast<int32_t>(std::round(std::sqrt(parallelHint)));
    Vector2ui imgRegion = Vector2ui(t->Extents());
    // Add some tolerance (%30)
    tileHint = uint32_t(Float(0.3) * Float(tileHint));
    Vector2ui tileSize = Vector2ui(FindOptimumTile(imgRegion[0], tileHint),
                                   FindOptimumTile(imgRegion[1], tileHint));
    renderBuffer->Resize(tileSize, 1, 3);
    tileCount = MathFunctions::DivideUp(imgRegion, tileSize);

    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    curFramebufferSize = Vector2ui(t->Extents());
    curFBMin = Vector2ui::Zero();
    curFBMax = curFramebufferSize;

    return renderBuffer->GetBufferInfo(curColorSpace,
                                       curFramebufferSize, 1);
}

RendererOutput TexViewRenderer::DoRender()
{
    // Use CPU timer here
    // TODO: Implement a GPU timer later
    Timer timer;
    timer.Start();

    const GPUDevice& device = gpuSystem.BestDevice();

    if(textures.empty()) return {};

    // Determine the current tile size
    using MathFunctions::Roll;
    uint32_t tileIndex = Roll<int32_t>(curTileIndex, 0,  tileCount.Multiply());
    Vector2ui tileIndex2D = Vector2ui(tileIndex % tileCount[0],
                                      tileIndex / tileCount[0]);
    Vector2ui regionMin = tileIndex2D * renderBuffer->Extents();
    Vector2ui regionMax = (regionMin + 1) * renderBuffer->Extents();
    regionMax.Clamp(Vector2ui::Zero(), curFramebufferSize);

    using namespace std::string_view_literals;
    const GPUQueue& processQueue = device.GetComputeQueue(0);
    Span<Float> pixels = renderBuffer->Pixels();
    Span<Float> samples = renderBuffer->Samples();

    Vector2ui curPixelCount2D = regionMax - regionMin;
    uint32_t curPixelCount = curPixelCount2D.Multiply();
    processQueue.IssueSaturatingKernel<KCColorTiles>
    (
        "KCColorTiles"sv,
        KernelIssueParams{.workCount = curPixelCount},
        //
        pixels,
        samples,
        tileIndex,
        curPixelCount2D,
        renderBuffer->ChannelCount()
    );

    // Send the resulting image to host (Acquire synchronization prims)
    const GPUQueue& transferQueue = device.GetTransferQueue();
    Optional<RenderImageSection>
    renderOut = renderBuffer->TransferToHost(processQueue,
                                             transferQueue);

    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value())
        return RendererOutput{};

    // Actually set the section parameters
    renderOut.value().pixelMin = regionMin;
    renderOut.value().pixelMax = regionMax;
    renderOut.value().globalWeight = Float(1);

    // Now wait, and send the information about timing etc.
    processQueue.Barrier().Wait();
    timer.Split();

    // Calculate M sample per sec
    double timeSec = timer.Elapsed<Second>();
    double samplePerSec = static_cast<double>(curPixelCount) / timeSec;
    samplePerSec /= 1'000'000;

    double spp = double(curTileIndex) / double(tileCount.Multiply());
    const CommonTexture* curTex = textures[textureIndex];

    curTileIndex++;
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            samplePerSec,
            "M samples/s",
            spp,
            "spp",
            float(timer.Elapsed<Millisecond>()),
            Vector2ui(curTex->Extents()),
            MRayColorSpaceEnum::MR_ACES_CG,
            static_cast<uint32_t>(textures.size()),
            static_cast<uint32_t>(curTex->MipCount())
        },
        .imageOut = renderOut
    };
}

void TexViewRenderer::StopRender()
{}