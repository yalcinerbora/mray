#include "TexViewRenderer.h"
#include "TracerBase.h"
#include "TextureView.hpp"

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
void KCColorTiles(MRAY_GRID_CONSTANT const ImageSpan imgSpan,
                  MRAY_GRID_CONSTANT const uint32_t colorIndex)
{
    KernelCallParams kp;

    uint32_t totalPix = imgSpan.Extent().Multiply();
    Vector3 color = Color::RandomColorRGB(colorIndex);
    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        Vector2i pixelIndex = imgSpan.LinearIndexTo2D(int32_t(i));
        imgSpan.StorePixel(color, pixelIndex);
        imgSpan.StoreWeight(Float(1), pixelIndex);
    }
}

template<uint32_t C>
MRAY_KERNEL
void KCShowTexture(MRAY_GRID_CONSTANT const ImageSpan imgSpan,
                   MRAY_GRID_CONSTANT const Vector2ui tileStart,
                   MRAY_GRID_CONSTANT const Vector2ui texResolution,
                   MRAY_GRID_CONSTANT const uint32_t mipIndex,
                   MRAY_GRID_CONSTANT const GenericTextureView texView)
{
    KernelCallParams kp;

    Vector2i regionSize = imgSpan.Extent();
    int32_t totalPix = regionSize.Multiply();
    // Loop over the output span
    for(int32_t i = int32_t(kp.GlobalId());
        i < totalPix; i += int32_t(kp.TotalSize()))
    {
        // Calculate uv coords
        Vector2i localIndexInt = imgSpan.LinearIndexTo2D(i);
        Vector2 localIndex = Vector2(localIndexInt);
        Vector2 globalIndex = Vector2(tileStart) + localIndex + Vector2(0.5);
        Vector2 uv = globalIndex / Vector2(texResolution);

        Vector3 result;
        if constexpr(C == 1)
        {
            // For single channel textures, show grayscale
            const auto& view = std::get<TracerTexView<2, Float>>(texView);
            result = Vector3(view(uv, Float(mipIndex)));
        }
        else if constexpr(C == 2)
        {
            const auto& view = std::get<TracerTexView<2, Vector<C, Float>>>(texView);
            result = Vector3(view(uv, Float(mipIndex)), 0);
        }
        else if constexpr(C == 3)
        {
            const auto& view = std::get<TracerTexView<2, Vector<C, Float>>>(texView);
            result = Vector3(view(uv, Float(mipIndex)));
        }
        else if constexpr(C == 4)
        {
            // Drop the last pixel
            // TODO: change this to something else,
            // We drop the last pixel since it will be considered as alpha
            // if we send it to visor and it may be invisible
            const auto& view = std::get<TracerTexView<2, Vector<C, Float>>>(texView);
            result = Vector3(view(uv, Float(mipIndex)));
        }
        imgSpan.StoreWeight(Float(1), localIndexInt);
        imgSpan.StorePixel(result, localIndexInt);
    }
}

TexViewRenderer::TexViewRenderer(const RenderImagePtr& rb,
                                 TracerView tv,
                                 ThreadPool& tp,
                                 const GPUSystem& s,
                                 const RenderWorkPack& wp)
    : RendererT(rb, wp, tv, s, tp)
    , saveImage(true)
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
                                              CamSurfaceId,
                                              uint32_t customLogicIndex0,
                                              uint32_t customLogicIndex1)
{
    // TODO: This is common assignment, every renderer
    // does this move to a templated intermediate class
    // on the inheritance chain
    currentOptions = newOptions;
    totalIterationCount = 0;

    // Skip if nothing to show
    if(textures.empty())
    {
        MRAY_WARNING_LOG("[(R)TexView]: No textures are present "
                         "in the tracer. Rendering nothing!");
        return RenderBufferInfo
        {
            .data = nullptr,
            .totalSize = 0,
            .renderColorSpace = tracerView.tracerParams.globalTextureColorSpace,
            .resolution = Vector2ui::Zero()
        };
    }

    // Find the texture index
    using Math::Roll;
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
    Vector2ui mipSize = Graphics::TextureMipSize(Vector2ui(t->Extents()), mipIndex);

    RenderImageParams rIParams
    {
        .resolution = mipSize,
        .regionMin = Vector2ui::Zero(),
        .regionMax = mipSize,
    };
    imageTiler = ImageTiler(renderBuffer.get(), rIParams,
                            tracerView.tracerParams.parallelizationHint,
                            Vector2ui::Zero());

    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    auto bufferPtrAndSize = renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = curColorSpace,
        .resolution = imageTiler.FullResolution(),
        .curRenderLogic0 = textureIndex,
        .curRenderLogic1 = mipIndex
    };
}

RendererOutput TexViewRenderer::DoRender()
{
    static const auto annotation = gpuSystem.CreateAnnotation("Render Frame");
    const auto _ = annotation.AnnotateScope();

    // Use CPU timer here
    // TODO: Implement a GPU timer later
    Timer timer; timer.Start();
    const GPUDevice& device = gpuSystem.BestDevice();

    if(textures.empty()) return {};

    using namespace std::string_view_literals;
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    uint32_t curPixelCount = imageTiler.CurrentTileSize().Multiply();
    // Do not start writing to device side untill copy is complete
    // (device buffer is read fully)
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    switch(currentOptions.mode)
    {
        case Mode::SHOW_TILING:
        {
            processQueue.IssueWorkKernel<KCColorTiles>
            (
                "KCColorTiles"sv,
                DeviceWorkIssueParams{.workCount = curPixelCount},
                //
                imageTiler.GetTileSpan(),
                imageTiler.CurrentTileIndex1D()
            );
            break;
        }
        case Mode::SHOW_TEXTURES:
        {
            auto texView = *textureViews[textureIndex];
            uint32_t texChannelCount = FindTexViewChannelCount(texView);
            auto KernelCall = [&, this]<uint32_t C>()
            {
                // Do not start writing to device side untill copy is complete
                // (device buffer is read fully)
                processQueue.IssueWorkKernel<KCShowTexture<C>>
                (
                    "KCShowTexture"sv,
                    DeviceWorkIssueParams{.workCount = curPixelCount},
                    //
                    imageTiler.GetTileSpan(),
                    imageTiler.LocalTileStart(),
                    imageTiler.FullResolution(),
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
    renderOut = imageTiler.TransferToHost(processQueue,
                                          transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value())
        return RendererOutput{};

    // Actually set the section parameters
    renderOut->globalWeight = Float(1);
    // Now wait, and send the information about timing etc.
    processQueue.Barrier().Wait();
    timer.Split();

    // Calculate M sample per sec
    double timeSec = timer.Elapsed<Second>();
    double samplePerSec = static_cast<double>(curPixelCount) / timeSec;
    samplePerSec /= 1'000'000;
    double spp = double(1) / double(imageTiler.TileCount().Multiply());
    totalIterationCount++;
    spp *= static_cast<double>(totalIterationCount);

    bool triggerSave = saveImage &&
        (currentOptions.totalSPP * imageTiler.TileCount().Multiply() ==
         totalIterationCount);
    // Roll to the next tile
    imageTiler.NextTile();
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            samplePerSec,
            "M pixels/s",
            spp,
            double(currentOptions.totalSPP),
            "pix",
            float(timer.Elapsed<Millisecond>()),
            imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            static_cast<uint32_t>(textures.size()),
            static_cast<uint32_t>(textures[textureIndex]->MipCount())
        },
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

void TexViewRenderer::StopRender()
{}

std::string_view TexViewRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "TexView"sv;
    return RendererTypeName<Name>;
}

size_t TexViewRenderer::GPUMemoryUsage() const
{
    return 0;
}

