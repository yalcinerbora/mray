#include "TexViewRenderer.h"
#include "TextureView.hpp"  // IWYU pragma: keep
#include "SpectrumContext.h"

#include "Device/GPUSystem.hpp"

#include "Core/Timer.h"
#include "Core/ColorFunctions.h"

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

    uint32_t totalPix = uint32_t(imgSpan.Extent().Multiply());
    Vector3 color = Color::RandomColorRGB(colorIndex);
    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        Vector2i pixelIndex = imgSpan.LinearIndexTo2D(i);
        imgSpan.StorePixel(color, pixelIndex);
        imgSpan.StoreWeight(Float(1), pixelIndex);
    }
}

template<uint32_t C>
MRAY_KERNEL
void KCShowTexture(// Output
                   MRAY_GRID_CONSTANT const ImageSpan imgSpan,
                   // Constants
                   MRAY_GRID_CONSTANT const Vector2ui tileStart,
                   MRAY_GRID_CONSTANT const Vector2ui texResolution,
                   MRAY_GRID_CONSTANT const uint32_t mipIndex,
                   MRAY_GRID_CONSTANT const GenericTextureView texView)
{
    KernelCallParams kp;

    Vector2i regionSize = imgSpan.Extent();
    uint32_t totalPix = uint32_t(regionSize.Multiply());
    // Loop over the output span
    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        // Calculate uv coords
        Vector2i localIndexInt = imgSpan.LinearIndexTo2D(uint32_t(i));
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

template<uint32_t C>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSampleTextureSpectral(// I-O
                             MRAY_GRID_CONSTANT const Span<Spectrum> dThroughput,
                             // Input
                             MRAY_GRID_CONSTANT const Span<const SpectrumWaves> dWavelengths,
                             // We need the tile info from the span we do not actually
                             // use the ptrs.
                             MRAY_GRID_CONSTANT const ImageSpan imgSpan,
                             // Constants
                             MRAY_GRID_CONSTANT const Vector2ui tileStart,
                             MRAY_GRID_CONSTANT const Vector2ui texResolution,
                             MRAY_GRID_CONSTANT const uint32_t mipIndex,
                             MRAY_GRID_CONSTANT const GenericTextureView texView,
                             MRAY_GRID_CONSTANT const Jakob2019Detail::Data data,
                             MRAY_GRID_CONSTANT const bool isIlluminant)
{
    KernelCallParams kp;
    Vector2i regionSize = imgSpan.Extent();
    uint32_t totalPix = uint32_t(regionSize.Multiply());
    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        // Calculate uv coords
        Vector2i localIndexInt = imgSpan.LinearIndexTo2D(i);
        Vector2 localIndex = Vector2(localIndexInt);
        Vector2 globalIndex = Vector2(tileStart) + localIndex + Vector2(0.5);
        Vector2 uv = globalIndex / Vector2(texResolution);

        // TODO: More texture types
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

        // Convert to the spectrum
        using Converter = typename SpectrumContextJakob2019::Converter;
        SpectrumWaves waves = dWavelengths[i];
        auto converter = Converter(waves, data);
        Spectrum s = isIlluminant ? converter.ConvertRadiance(result)
                                  : converter.ConvertAlbedo(result);
        Spectrum t = dThroughput[i];

        // Multiply with the "1 / PDF"
        // (we do not explicitly store PDFs yet (~32MiB extra memory))
        t = t * s;

        // Multiply with Illuminant, since these colors are optimized by
        // that illum conditions.
        // This means we do furnace test, and it simulates as if it hits a D65 "light".
        static constexpr Float OFFSET = Float(0.5) - Float(Color::CIE_1931_RANGE[0]);
        MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
        for(uint32_t j = 0; j < SpectraPerSpectrum; j++)
            t[j] *= data.spdIlluminant(waves[j] + OFFSET);

        dThroughput[i] = t;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSplatRGBToImageSpan(MRAY_GRID_CONSTANT const ImageSpan imgSpan,
                           MRAY_GRID_CONSTANT const Span<const Spectrum> dSpectrumAsRGB)
{
    KernelCallParams kp;

    Vector2i regionSize = imgSpan.Extent();
    uint32_t totalPix = uint32_t(regionSize.Multiply());
    // Loop over the output span
    for(uint32_t i = kp.GlobalId(); i < totalPix; i += kp.TotalSize())
    {
        Vector2i localIndexInt = imgSpan.LinearIndexTo2D(i);
        imgSpan.StoreWeight(Float(1), localIndexInt);
        imgSpan.StorePixel(Vector3(dSpectrumAsRGB[i]), localIndexInt);
    }
}

void TexViewRenderer::RenderTextureAsData(const GPUQueue& processQueue)
{
    using namespace std::string_view_literals;

    uint32_t curPixelCount = imageTiler.CurrentTileSize().Multiply();
    auto texView = *textureViews[textureIndex];
    uint32_t texChannelCount = FindTexViewChannelCount(texView);
    auto KernelCall = [&, this]<uint32_t C>()
    {
        // Do not start writing to device side until copy is complete
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
}

void TexViewRenderer::RenderTextureAsSpectral(const GPUQueue& processQueue)
{
    rnGenerator->SetupRange(this->imageTiler.LocalTileStart(),
                            this->imageTiler.LocalTileEnd(),
                            processQueue);

    RNRequestList perSampleRNCountList = spectrumContext->SampleSpectrumRNList();
    uint32_t curPixelCount = imageTiler.CurrentTileSize().Multiply();
    uint32_t totalRNCount = curPixelCount * perSampleRNCountList.TotalRNCount();
    auto dRandomNumbersLocal = dRandomNumbers.subspan(0,  totalRNCount);
    auto dThroughputLocal = dThroughputs.subspan(0, curPixelCount);
    auto dWavelengthsLocal = dWavelengths.subspan(0, curPixelCount);
    auto dWavelengthPDFsLocal = dWavelengthPDFs.subspan(0, curPixelCount);
    // Generate random numbers
    rnGenerator->GenerateNumbers(dRandomNumbers, 0, perSampleRNCountList,
                                 processQueue);
    // Sample spectrum
    spectrumContext->SampleSpectrumWavelengths(dWavelengths, dWavelengthPDFsLocal,
                                               dRandomNumbers, processQueue);

    // Sample texture
    auto texView = *textureViews[textureIndex];
    bool isIlluminant = (textures[textureIndex]->IsIlluminant() ==
                         MRayTextureIsIlluminant::IS_ILLUMINANT);
    uint32_t texChannelCount = FindTexViewChannelCount(texView);
    using namespace std::string_view_literals;
    auto KernelCall = [&, this]<uint32_t C>()
    {
        processQueue.IssueWorkKernel<KCSampleTextureSpectral<C>>
        (
            "KCSampleTextureSpectral"sv,
            DeviceWorkIssueParams{.workCount = curPixelCount},
            //
            dThroughputLocal,
            dWavelengthsLocal,
            //
            imageTiler.GetTileSpan(),
            imageTiler.LocalTileStart(),
            imageTiler.FullResolution(),
            mipIndex,
            texView,
            spectrumContext->GetData(),
            isIlluminant
        );
    };
    switch(texChannelCount)
    {
        case 1: KernelCall.template operator()<1>(); break;
        case 2: KernelCall.template operator()<2>(); break;
        case 3: KernelCall.template operator()<3>(); break;
        case 4: KernelCall.template operator()<4>(); break;
    }

    // Convert to RGB
    spectrumContext->ConvertSpectrumToRGB(dThroughputLocal, dWavelengthsLocal,
                                          dWavelengthPDFsLocal, processQueue);
    // Write these RGB to buffer
    processQueue.IssueWorkKernel<KCSplatRGBToImageSpan>
    (
        "KCSplatRGBToImageSpan"sv,
        DeviceWorkIssueParams{.workCount = curPixelCount},
        //
        imageTiler.GetTileSpan(),
        ToConstSpan(dThroughputLocal)
    );
}

TexViewRenderer::TexViewRenderer(const RenderImagePtr& rb,
                                 TracerView tv,
                                 ThreadPool& tp,
                                 const GPUSystem& s,
                                 const RenderWorkPack& wp)
    : RendererT(rb, wp, tv, s, tp)
    , saveImage(true)
    , spectrumMem(gpuSystem.AllGPUs(), 4_MiB, 32_MiB, true)
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
    return StaticAttributeInfo();
}

RendererOptionPack TexViewRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<bool>{}, 1));
    result.attributes.back().Push(Span<const bool>(&currentOptions.isSpectral, 1));

    if constexpr(MRAY_IS_DEBUG)
    {
        for([[maybe_unused]] const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void TexViewRenderer::PushAttribute(uint32_t attributeIndex,
                                    TransientData data, const GPUQueue&)
{
    switch(attributeIndex)
    {
        case 0: newOptions.totalSPP = data.AccessAs<uint32_t>()[0]; break;
        case 1: newOptions.isSpectral = data.AccessAs<bool>()[0];   break;
        //
        default: throw MRayError("{} Unknown attribute index {}",
                                 TypeName(), attributeIndex);
    }
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
    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
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

    // Setup Image Tiler
    RenderImageParams rIParams
    {
        .resolution = mipSize,
        .regionMin = Vector2ui::Zero(),
        .regionMax = mipSize,
    };
    imageTiler = ImageTiler(renderBuffer.get(), rIParams,
                            tracerView.tracerParams.parallelizationHint,
                            Vector2ui::Zero());



    // Load spectral system if spectral mode is enabled
    if(currentOptions.isSpectral && t->IsColor() == AttributeIsColor::IS_COLOR)
    {
        // Don't bother reloading context if colorspace is same
        if(!spectrumContext || spectrumContext->ColorSpace() != curColorSpace)
        {
            using SC = SpectrumContextJakob2019;
            spectrumContext = std::make_unique<SC>(curColorSpace,
                                                   tracerView.tracerParams.wavelengthSampleMode,
                                                   gpuSystem);
        }

        // Allocate sampling buffers
        uint32_t maxRayCount = this->imageTiler.ConservativeTileSize().Multiply();
        uint32_t sampleRNCount = spectrumContext->SampleSpectrumRNList().TotalRNCount();
        uint32_t maxRNCount = sampleRNCount * maxRayCount;
        MemAlloc::AllocateMultiData(Tie(dThroughputs, dWavelengths,
                                        dWavelengthPDFs, dRandomNumbers),
                                    spectrumMem,
                                    {maxRayCount, maxRayCount,
                                     maxRayCount, maxRNCount});
        // Finally allocate RNG
        auto RngGen = tracerView.rngGenerators.at(tracerView.tracerParams.samplerType.e);
        if(!RngGen)
            throw MRayError("[{}]: Unknown random number generator type {}.", TypeName(),
                            uint32_t(tracerView.tracerParams.samplerType.e));
        uint64_t seed = this->tracerView.tracerParams.seed;
        Vector2ui maxDeviceLocalRNGCount = this->imageTiler.ConservativeTileSize();
        rnGenerator = RngGen->get()(rIParams,
                                    std::move(maxDeviceLocalRNGCount),
                                    std::move(currentOptions.totalSPP), 
                                    std::move(seed), gpuSystem, 
                                    globalThreadPool);
    }

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

    // Render nothing...
    if(textures.empty()) return {};

    using namespace std::string_view_literals;
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    uint32_t curPixelCount = imageTiler.CurrentTileSize().Multiply();
    // Do not start writing to device side until copy is complete
    // (device buffer is read fully)
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    switch(currentOptions.renderMode)
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
            using enum AttributeIsColor;
            const auto* t = textures[textureIndex];
            bool doSpectral = (currentOptions.isSpectral && t->IsColor() == IS_COLOR);
            //
            if(doSpectral)  RenderTextureAsSpectral(processQueue);
            else            RenderTextureAsData(processQueue);

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

typename TexViewRenderer::AttribInfoList
TexViewRenderer::StaticAttributeInfo()
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP", MRayDataTypeRT(MR_UINT32), IS_SCALAR, MR_MANDATORY},
        {"isSpectral", MRayDataTypeRT(MR_BOOL), IS_SCALAR, MR_MANDATORY}
    };
}

size_t TexViewRenderer::GPUMemoryUsage() const
{
    size_t allSize = spectrumMem.Size();
    if(rnGenerator) allSize += rnGenerator->GPUMemoryUsage();
    if(spectrumContext) allSize += spectrumContext->GPUMemoryUsage();

    return allSize;
}

