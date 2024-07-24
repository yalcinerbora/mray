#include "TexViewRenderer.h"
#include "TracerBase.h"

#include "Device/GPUSystem.hpp"

#include "Core/Timer.h"
#include "Core/DeviceVisit.h"
#include "Core/ColorFunctions.h"

#include "Device/GPUAlgGeneric.h"

uint32_t FindTexViewChannelCount(const GenericTextureView& genericTexView)
{
    return std::visit([](auto&& v)
    {
        return std::remove_cvref_t<decltype(v)>::Channels;
    }, genericTexView);
}

MRAY_KERNEL
void KCColorTiles(MRAY_GRID_CONSTANT const Span<Float> dPixels,
                  MRAY_GRID_CONSTANT const Span<Float> dSamples,
                  MRAY_GRID_CONSTANT const uint32_t colorIndex,
                  MRAY_GRID_CONSTANT const Vector2ui resolution,
                  MRAY_GRID_CONSTANT const uint32_t channelCount)
{
    KernelCallParams kp;

    uint32_t totalPix = resolution.Multiply();
    assert(totalPix * channelCount <= dPixels.size());
    assert(totalPix <= dSamples.size());

    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        uint32_t pixIndex = i * channelCount;
        uint32_t sampleIndex = i;
        Vector3 color = Color::RandomColorRGB(colorIndex);

        dSamples[sampleIndex] = Float(1);
        dPixels[pixIndex + 0] = color[0];
        dPixels[pixIndex + 1] = color[1];
        dPixels[pixIndex + 2] = color[2];
    }
}

template<uint32_t C>
MRAY_KERNEL
void KCShowTexture(MRAY_GRID_CONSTANT const Span<Float> dPixels,
                   MRAY_GRID_CONSTANT const Span<Float> dSamples,
                   MRAY_GRID_CONSTANT const Vector2ui regionMin,
                   MRAY_GRID_CONSTANT const Vector2ui regionMax,
                   MRAY_GRID_CONSTANT const Vector2ui resolution,
                   MRAY_GRID_CONSTANT const uint32_t mipIndex,
                   MRAY_GRID_CONSTANT const GenericTextureView texView)
{
    KernelCallParams kp;

    Vector2ui regionSize = regionMax - regionMin;
    uint32_t totalPix = regionSize.Multiply();
    assert(totalPix * 3 <= dPixels.size());
    assert(totalPix <= dSamples.size());

    for(uint32_t wIndexLinear = kp.GlobalId(); wIndexLinear < totalPix;
        wIndexLinear += kp.TotalSize())
    {
        // Calculate uv coords
        Vector2 wIndex2D = Vector2(wIndexLinear % regionSize[0],
                                   wIndexLinear / regionSize[0]);
        Vector2 rIndex2D = Vector2(regionMin) + wIndex2D + Vector2(0.5);
        Vector2 uv = rIndex2D / Vector2(resolution);

        Vector3 result;
        if constexpr(C == 1)
        {
            const auto& view = std::get<TracerTexView<2, Float>>(texView);
            result = Vector3(view(uv, Float(mipIndex)).value(), 0, 0);
        }
        else if constexpr(C == 2)
        {
            const auto& view = std::get<TracerTexView<2, Vector<C, Float>>>(texView);
            result = Vector3(view(uv, Float(mipIndex)).value(), 0);
        }
        else if constexpr(C == 3)
        {
            const auto& view = std::get<TracerTexView<2, Vector<C, Float>>>(texView);
            result = Vector3(view(uv, Float(mipIndex)).value());
        }
        else if constexpr(C == 4)
        {
            // Drop the last pixel
            // TODO: change this to something else,
            // We drop the last pixel since it will be considered as alpha
            // if we send it to visor
            const auto& view = std::get<TracerTexView<2, Vector<C, Float>>>(texView);
            result = Vector3(view(uv, Float(mipIndex)).value());
        }

        uint32_t wIndexFloat = wIndexLinear * 3;
        dSamples[wIndexLinear] = Float(1);
        dPixels[wIndexFloat + 0] = result[0];
        dPixels[wIndexFloat + 1] = result[1];
        dPixels[wIndexFloat + 2] = result[2];
    }
}

TexViewRenderer::TexViewRenderer(const RenderImagePtr& rb,
                                 TracerView tv, const GPUSystem& s)
    : RendererT(rb, tv, s)
{
    // Pre-generate list
    textures.clear();
    textureViews.clear();
    //
    textures.reserve(tracerView.textures.size());
    textureViews.reserve(tracerView.textures.size());
    for(const auto& [texId, tex] : tracerView.textures)
    {
        // Skip 1D/3D textures we can not render those
        if(tex.DimensionCount() != 2) continue;
        textures.push_back(&tex);
        const auto& tView = tracerView.textureViews.at(texId).value().get();
        textureViews.push_back(&tView);
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

    if constexpr(MRAY_IS_DEBUG)
    {
        for(const auto& d: result.attributes)
            assert(d.IsFull());
    }
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
    if(textures.empty())
    {
        MRAY_WARNING_LOG("[(R)TexView] No textures are present "
                         "in the tracer. Rendering nothing!");
        return RenderBufferInfo
        {
            .data = nullptr,
            .totalSize = 0,
            .renderColorSpace = tracerView.tracerParams.globalTextureColorSpace,
            .resolution = Vector2ui::Zero(),
            .depth = 0
        };
    }

    // Find the texture index
    using MathFunctions::Roll;
    uint32_t newTextureIndex = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                             int32_t(textures.size())));
    const GenericTexture* t = textures[newTextureIndex];
    // Mip Index
    // Change to zero if texture is changed
    mipIndex = (newTextureIndex == textureIndex)
                ? uint32_t(Roll(int32_t(customLogicIndex1), 0,
                                int32_t(t->MipCount())))
                : 0;
    textureIndex = newTextureIndex;
    // And mip size
    mipSize = Graphics::TextureMipSize(Vector2ui(t->Extents()), mipIndex);
    // Initialize tile index
    curTileIndex = 0;

    // Calculate tile size according to the parallelization hint
    uint32_t parallelHint = tracerView.tracerParams.parallelizationHint;
    Vector2ui imgRegion = mipSize;
    Vector2ui tileSize = FindOptimumTile(imgRegion, parallelHint);
    renderBuffer->Resize(tileSize, 1, 3);
    tileCount = MathFunctions::DivideUp(imgRegion, tileSize);

    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    curFramebufferSize = mipSize;
    curFBMin = Vector2ui::Zero();
    curFBMax = curFramebufferSize;

    RenderBufferInfo rbI = renderBuffer->GetBufferInfo(curColorSpace,
                                                       curFramebufferSize, 1);
    rbI.curRenderLogic0 = textureIndex;
    rbI.curRenderLogic1 = mipIndex;
    return rbI;
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
    Vector2ui regionMax = (tileIndex2D + 1) * renderBuffer->Extents();
    regionMax.Clamp(Vector2ui::Zero(), curFramebufferSize);

    using namespace std::string_view_literals;
    const GPUQueue& processQueue = device.GetComputeQueue(0);
    Vector2ui curPixelCount2D = regionMax - regionMin;
    uint32_t curPixelCount = curPixelCount2D.Multiply();

    // Do not start writing to device side untill copy is complete
    // (device buffer is read fully)
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    switch(currentOptions.mode)
    {
        case Mode::SHOW_TILING:
        {
            processQueue.IssueSaturatingKernel<KCColorTiles>
            (
                "KCColorTiles"sv,
                KernelIssueParams{.workCount = curPixelCount},
                //
                renderBuffer->Pixels(),
                renderBuffer->Samples(),
                tileIndex,
                curPixelCount2D,
                renderBuffer->ChannelCount()
            );
            break;
        }
        case Mode::SHOW_TEXTURES:
        {
            auto texView = *textureViews[textureIndex];
            uint32_t texChannelCount = FindTexViewChannelCount(texView);
            Vector2ui resolution = mipSize;
            auto KernelCall = [&, this]<uint32_t C>()
            {
                // Do not start writing to device side untill copy is complete
                // (device buffer is read fully)
                processQueue.IssueSaturatingKernel<KCShowTexture<C>>
                (
                    "KCShowTexture"sv,
                    KernelIssueParams{.workCount = curPixelCount},
                    //
                    renderBuffer->Pixels(),
                    renderBuffer->Samples(),
                    regionMin,
                    regionMax,
                    resolution,
                    mipIndex,
                    texView
                );
            };
            switch(texChannelCount)
            {
                case 1: KernelCall.template operator()<1>(); break;
                case 2: KernelCall.template operator()<2>(); break;
                case 3: KernelCall.template operator()<3>(); break;
                case 4: KernelCall.template operator()<4>(); break;
            }
            break;
        }
    }

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

    double spp = double(curTileIndex + 1) / double(tileCount.Multiply());
    const GenericTexture* curTex = textures[textureIndex];

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
            mipSize,
            MRayColorSpaceEnum::MR_ACES_CG,
            static_cast<uint32_t>(textures.size()),
            static_cast<uint32_t>(curTex->MipCount())
        },
        .imageOut = renderOut
    };
}

void TexViewRenderer::StopRender()
{}