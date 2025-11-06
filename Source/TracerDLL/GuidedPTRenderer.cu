#include "GuidedPTRenderer.h"

#include "Core/MemAlloc.h"
#include "Core/Timer.h"
#include "Tracer/RendererCommon.h"

struct TransientMCState
{
    GuidedPTRDetail::MCState s;
    uint32_t N;
};

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCAccumulateShadowRaysGPT(MRAY_GRID_CONSTANT const Span<Spectrum> dRadianceOut,
                               MRAY_GRID_CONSTANT const Span<const Spectrum> dShadowRayRadiance,
                               MRAY_GRID_CONSTANT const Bitspan<const uint32_t> dIsVisibleBuffer,
                               MRAY_GRID_CONSTANT const Span<const PathDataPack> dPathDataPack)
{
    KernelCallParams kp;
    uint32_t shadowRayCount = static_cast<uint32_t>(dShadowRayRadiance.size());
    for(uint32_t i = kp.GlobalId(); i < shadowRayCount; i += kp.TotalSize())
    {
        PathDataPack dataPack = dPathDataPack[i];
        bool isDead = (dataPack.status[uint32_t(PathStatusEnum::DEAD)] ||
                       dataPack.status[uint32_t(PathStatusEnum::INVALID)]);
        bool isSpecular = dataPack.type == RayType::SPECULAR_RAY;
        bool isVisible = dIsVisibleBuffer[i];
        if(!isVisible || isSpecular || isDead) continue;
        //
        dRadianceOut[i] += dShadowRayRadiance[i];
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCInitGuidedPathStatesIndirect(MRAY_GRID_CONSTANT const Span<uint32_t> dLiftedMarkovChainIndex,
                                    MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    KernelCallParams kp;
    uint32_t newPathCount = uint32_t(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < newPathCount; i += kp.TotalSize())
    {
        dLiftedMarkovChainIndex[dIndices[i]] = GuidedPTRDetail::INVALID_MC_INDEX;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCWriteHashGridDataToImg(MRAY_GRID_CONSTANT const ImageSpan img,
                              // I-O
                              MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                              // Input
                              MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                              MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                              MRAY_GRID_CONSTANT const Span<const Float> dFilterWeights,
                              MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                              // Constants
                              //MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState gs,
                              MRAY_GRID_CONSTANT const HashGridView hashGrid,
                              MRAY_GRID_CONSTANT const Span<GuidedPTRDetail::MCState> dMCStates,
                              MRAY_GRID_CONSTANT const Span<GuidedPTRDetail::MCCount> dMCCounts,
                              MRAY_GRID_CONSTANT const Span<GuidedPTRDetail::MCIrradiance> dMCIrradiances,
                              MRAY_GRID_CONSTANT const GuidedPTRDetail::DisplayMode::E displayMode)
{
    using namespace GuidedPTRDetail;

    uint32_t sampleCount = static_cast<uint32_t>(dIndices.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        uint32_t index = dIndices[i];
        Float weight = dFilterWeights[index];
        Vector2i pixCoords = Vector2i(dImgCoords[index].pixelIndex);
        BackupRNG rng(dRNGStates[index]);

        auto [ray, tMM] = RayFromGMem(dRays, index);
        Vector3 pos = ray.AdvancedPos(tMM[1]);
        Vector3 dir = Math::Normalize(-ray.dir);

        auto code = hashGrid.GenCodeStochastic(pos, dir, rng);
        auto loc = hashGrid.Search(code);
        if(!loc.has_value())
        {
            // Not having a 1st bounce cache is also an error
            // distinguish it with a cyan value
            // magenta is used for NaNs
            //img.AddToPixelAtomic(BIG_CYAN(), pixCoords);
            //img.AddToWeightAtomic(Float(128) * weight, pixCoords);
            //
            // Above comment were true when sampling were not stochastic
            // we ignore if we do not find a location
            img.AddToPixelAtomic(Vector3::Zero(), pixCoords);
            img.AddToWeightAtomic(Float(0), pixCoords);
            continue;
        }

        //
        Vector3 value;
        uint32_t hgIndex = *loc;
        switch(displayMode)
        {
            using enum DisplayMode::E;
            case GRID:
            {
                value = Color::RandomColorRGB(hgIndex);
                break;
            }
            case LIGHT_CACHE:
            {
                Float irrad = dMCIrradiances[hgIndex].irrad;
                value = Vector3(irrad);
                break;
            }
            case MC_DIR:
            {
                MCState state = dMCStates[hgIndex];
                uint32_t N = dMCCounts[hgIndex];
                value = SufficientStatsToLobe(state, N, pos).dir;
                value = (value + Vector3(1)) * Vector3(0.5);
                break;
            }
            case MC_IRRAD:
            {
                value = Vector3(dMCStates[hgIndex].weight);
                break;
            }
            case MC_COS:
            {
                MCState state = dMCStates[hgIndex];
                if(state.weight <= Float(0))
                {
                    value = Vector3::Zero();
                    break;
                }
                // [0, pi]
                Float invCos = Math::ArcCos(state.cos / state.weight);
                assert(Math::IsFinite(state.cos / state.weight));
                // [0, 1]
                invCos *= MathConstants::InvPi<Float>();
                value = Vector3(invCos);
                break;
            }
            // Should not be here
            case RENDER: default: break;
        }
        if(!Math::IsFinite(value))
        {
            value = BIG_MAGENTA();
            weight *= Float(128.0);
        }
        img.AddToPixelAtomic(value, pixCoords);
        img.AddToWeightAtomic(weight, pixCoords);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCBackpropagateHashGridPath(// Output
                                 MRAY_GRID_CONSTANT const Span<TransientMCState> dWriteMCStates,
                                 // I-O
                                 MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState globalState,
                                 MRAY_GRID_CONSTANT const Span<uint32_t> dLiftedMCIndices,
                                 MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                                 // Input
                                 MRAY_GRID_CONSTANT const Span<const Float> dPrevPathReflectance,
                                 MRAY_GRID_CONSTANT const Span<const Float> dScoreSums,
                                 // Determining if this is intermediate path (not hit light etc.)
                                 MRAY_GRID_CONSTANT const Span<const HitKeyPack> dHitKeys,
                                 MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                                 MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    using namespace GuidedPTRDetail;

    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t rIndex = dRayIndices[i];
        uint32_t isLightFlag = dHitKeys[rIndex].lightOrMatKey.FetchFlagPortion();
        uint32_t prevMCIndex = dLiftedMCIndices[rIndex];
        BackupRNG rng = BackupRNG(dRNGStates[rIndex]);
        //
        if(isLightFlag == IS_LIGHT_KEY_FLAG) continue;
        if(prevMCIndex == INVALID_MC_INDEX) continue;

        // ======================= //
        // Irradiance Cache Update //
        // ======================= //
        auto [ray, tMM] = RayFromGMem(dRays, rIndex);
        Vector3 position = ray.AdvancedPos(tMM[1]);
        Vector3 direction = Math::Normalize(-ray.dir);

        const HashGridView& hg = globalState.hashGrid;
        auto code = hg.GenCodeStochastic(position, direction, rng);
        auto irradLoc = hg.Search(code);
        //
        if(!irradLoc) continue;
        // Race condition here!
        // We read while somebody writes to this specific location
        // maybe. It is fine as long as it is properly updated by writers.
        // Reading a stale value just hinders backpropogation speed maybe.
        //
        // TODO: Check if proper syncronization is good / bad etc.
        auto& dIrradHashGrid = globalState.dMCIrradiances;
        Float irrad = dIrradHashGrid[*irradLoc].irrad;
        Float refl = dPrevPathReflectance[rIndex];
        Float outRadiance = refl * irrad;
        MCIrradiance::AtomicEMA(dIrradHashGrid[prevMCIndex], outRadiance);

        Float weight = outRadiance;
        Float scoreSum = dScoreSums[rIndex];
        // Not as good as previous mixture, skip
        if(rng.NextFloat() * scoreSum > weight * MC_LOBE_COUNT)
            continue;

        // ========================= //
        //  Stochastic New Location  //
        // ========================= //
        Vector3 mcPos = ray.pos;
        Vector3 mcDir = Math::Normalize(ray.dir);
        // Try 1-2 times to find a location
        Optional<uint32_t> mcLoc;
        for(uint32_t _ = 0; _ < 2; _++)
        {
            auto mcCode = hg.GenCodeStochastic(mcPos, mcDir, rng);
            mcLoc = hg.Search(mcCode);
            if(mcLoc) break;
        }
        if(!mcLoc) continue;
        // Save new index for writing
        uint32_t newIndex = *mcLoc;
        dLiftedMCIndices[rIndex] = newIndex;

        // ========================= //
        //  Markov Chain Lock Update //
        // ========================= //
        // This kernel will be used as deterministic lock mechanism
        // where largest rIndex will be the winner
        // At the next kernel, winners will use
        // "dPrevPathReflectanceOrOutRadiance" value
        // (which is outRadiance atm) to do the actual update.
        //
        // We lose some data here (only a single winner will update the
        // markov chain). But paper also does this via race condition.
        // But that is non-determistic.
        uint32_t lock = DeviceAtomic::AtomicMax(globalState.dMCLocks[newIndex], rIndex + 1);
        // Skip chain update calculations, we did not get the lock
        // Here we could get false positives, this just prevents the extra work below
        // a little bit.
        if(lock > (rIndex + 1)) continue;

        // ====================== //
        //   Markov Chain Update  //
        // ====================== //
        Vector3 target = position;
        Vector3 pos = ray.pos;
        Vector3 dir = Math::Normalize(ray.dir);
        // If a thread is here we have the lock,
        // we can freely modify
        const auto& dStates = globalState.dMCStates;
        const auto& dCounts = globalState.dMCCounts;
        MCState s = dStates[prevMCIndex];
        MCCount count = dCounts[prevMCIndex];
        // Update routine
        static constexpr Float MIN_EMA_RATIO_MC = Float(0.01);
        static constexpr uint16_t MAX_SAMPLE_MC = uint16_t(4096);
        count = Math::Min<uint16_t>(count + 1, MAX_SAMPLE_MC);

        Float tMC = Math::Max(Float(1) / Float(count), MIN_EMA_RATIO_MC);
        s.weight = Math::Lerp(s.weight, weight, tMC);
        s.target = Math::Lerp(s.target, target * weight, tMC);
        //
        Vector3 targetPos = (s.weight < Float(0)) ? (s.target / s.weight) : s.target;
        Vector3 stateDir = Math::Normalize(targetPos - pos);
        Float newCos = Math::Max(Math::Dot(dir, stateDir), Float(0));
        //s.cos = Math::Min(Math::Lerp(s.cos, weight * newCos, tMC), s.weight);
        s.cos = Math::Min(Math::Lerp(s.cos, weight * newCos, tMC), Float(1));

        // Write back to intermediate buffer
        dWriteMCStates[rIndex].s = s;
        dWriteMCStates[rIndex].N = count;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCWriteMarkovChains(// I-O
                         MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState globalState,
                         MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                         // Input
                         MRAY_GRID_CONSTANT const Span<const TransientMCState> dWriteMCStates,
                         MRAY_GRID_CONSTANT const Span<const uint32_t> dMCWriteIndices,
                         MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                         MRAY_GRID_CONSTANT const Span<const HitKeyPack> dHitKeys)
{
    using namespace GuidedPTRDetail;

    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t rIndex = dRayIndices[i];
        uint32_t isLightFlag = dHitKeys[rIndex].lightOrMatKey.FetchFlagPortion();
        uint32_t writeMCIndex = dMCWriteIndices[rIndex];
        //
        if(isLightFlag == IS_LIGHT_KEY_FLAG) continue;
        if(writeMCIndex == INVALID_MC_INDEX) continue;
        if(globalState.dMCLocks[writeMCIndex] != (rIndex + 1)) continue;

        BackupRNG rng(dRNGStates[rIndex]);
        if(rng.NextFloat() < globalState.lobeProbability)
        {
            auto mcW = dWriteMCStates[rIndex].s;
            auto mcCountW = dWriteMCStates[rIndex].N;
            if(!Math::IsFinite(mcW.cos))
            {
                printf("MC Cos Not finite!\n");
                mcW.cos = Float(0);
            }
            if(!Math::IsFinite(mcW.target))
            {
                printf("MC Target Not finite!\n");
                mcW.target = Vector3::Zero();
            }
            if(!Math::IsFinite(mcW.weight))
            {
                printf("MC Weight Not finite!\n");
                mcW.weight = Float(0);
            }

            globalState.dMCStates[writeMCIndex] = mcW;
            globalState.dMCCounts[writeMCIndex] = uint16_t(mcCountW);
        }
        else
        {
            GuidedPTRDetail::MCState zeroMC =
            {
                .target = Vector3::Zero(),
                .cos = Float(0),
                .weight = Float(0),
            };
            globalState.dMCStates[writeMCIndex] = zeroMC;
            globalState.dMCCounts[writeMCIndex] = uint16_t(0);
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCBackpropagateIrradNEE(// I-O
                             MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState globalState,
                             // Indirect Inputs (accessed by ray Indices)
                             MRAY_GRID_CONSTANT const Span<const Float> dShadowPrevPathOutRadiance,
                             MRAY_GRID_CONSTANT const Span<const uint32_t> dLiftedMCIndices,
                             MRAY_GRID_CONSTANT const Span<const PathDataPack> dPathDataPack,
                             // Diret Input
                             MRAY_GRID_CONSTANT const Bitspan<const uint32_t> dIsVisibleBuffer)
{
    using namespace GuidedPTRDetail;
    assert(dShadowPrevPathOutRadiance.size() == dLiftedMCIndices.size());
    assert(dLiftedMCIndices.size() == dPathDataPack.size());

    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dShadowPrevPathOutRadiance.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t prevMCIndex = dLiftedMCIndices[i];
        PathDataPack dataPack = dPathDataPack[i];
        bool isVisible = dIsVisibleBuffer[i];
        bool isDead = (dataPack.status[uint32_t(PathStatusEnum::DEAD)] ||
                       dataPack.status[uint32_t(PathStatusEnum::INVALID)]);
        bool isSpecular = (dataPack.type == RayType::SPECULAR_RAY);
        bool hasNoPrev = (prevMCIndex == INVALID_MC_INDEX);
        //
        if(!isVisible || isSpecular || isDead || hasNoPrev) continue;
        //
        auto& dIrradHashGrid = globalState.dMCIrradiances;
        Float radEstimate = dShadowPrevPathOutRadiance[i];

        if(!Math::IsFinite(radEstimate))
        {
            printf("NEE Not finite!\n");
            radEstimate = Float(0);
        }

        MCIrradiance::AtomicEMA(dIrradHashGrid[prevMCIndex], radEstimate);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCBackpropagateIrradLight(// I-O
                               MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState globalState,
                               // Input
                               MRAY_GRID_CONSTANT const Span<const uint32_t> dLiftedMCIndices,
                               MRAY_GRID_CONSTANT const Span<const Float> dPrevPathReflectanceOrOutRadiance,
                               // Determining if this is intermediate path (not hit light etc.)
                               MRAY_GRID_CONSTANT const Span<const HitKeyPack> dHitKeys,
                               MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    using namespace GuidedPTRDetail;

    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t rIndex = dRayIndices[i];
        uint32_t isLightFlag = dHitKeys[rIndex].lightOrMatKey.FetchFlagPortion();
        uint32_t prevMCIndex = dLiftedMCIndices[rIndex];
        //
        if(isLightFlag != IS_LIGHT_KEY_FLAG) continue;
        if(prevMCIndex == INVALID_MC_INDEX) continue;

        // For this kernel
        // "dPrevPathReflectanceOrOutRadiance"
        // has full radiance estimate since the light work kernel
        // updated it.
        Float outRadiance = dPrevPathReflectanceOrOutRadiance[rIndex];


        if(!Math::IsFinite(outRadiance))
        {
            printf("Light Not finite!\n");
            outRadiance = Float(0);
        }

        auto& dIrradHashGrid = globalState.dMCIrradiances;
        MCIrradiance::AtomicEMA(dIrradHashGrid[prevMCIndex], outRadiance);
    }
}

GuidedPTRenderer::GuidedPTRenderer(const RenderImagePtr& rb,
                                   TracerView tv,
                                   ThreadPool& tp,
                                   const GPUSystem& s,
                                   const RenderWorkPack& wp)
    : Base(rb, tv, tp, s, wp, TypeName())
    , metaLightArray(s)
    , hashGrid(s)
    , rendererGlobalMem(s.AllGPUs(), 128_MiB, 1024_MiB)
    , saveImage(true)
{}

typename GuidedPTRenderer::AttribInfoList
GuidedPTRenderer::AttributeInfo() const
{
    return StaticAttributeInfo();
}

RendererOptionPack
GuidedPTRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    auto Push = [&result]<class T>(const T& in)
    {
        using LT = std::in_place_type_t<T>;
        result.attributes.push_back(TransientData(LT{}, 1));
        result.attributes.back().Push(Span<const T>(&in, 1));
    };
    auto PushEnum = [&result]<class T>(const T& in)
    {
        using LT = std::in_place_type_t<std::string_view>;
        std::string_view name = in.ToString();
        result.attributes.push_back(TransientData(LT{}, name.size()));
        auto buffer = result.attributes.back().AccessAsString();
        assert(buffer.size() == name.size());
        std::copy(name.cbegin(), name.cend(), buffer.begin());
    };

    Push(currentOptions.cacheEntryLimit);
    Push(currentOptions.cachePosBits);
    Push(currentOptions.cacheNormalBits);
    Push(currentOptions.cacheLevelCount);
    Push(currentOptions.cacheConeAperture);
    Push(currentOptions.russianRouletteRange);
    Push(currentOptions.totalSPP);
    PushEnum(currentOptions.lightSampler);
    PushEnum(currentOptions.renderMode);
    Push(currentOptions.lobeProbablity);
    PushEnum(currentOptions.displayMode);
    if constexpr(MRAY_IS_DEBUG)
    {
        for([[maybe_unused]] const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void GuidedPTRenderer::PushAttribute(uint32_t attributeIndex,
                                     TransientData data, const GPUQueue&)
{
    auto Load = []<class T>(T & out, const TransientData & data)
    {
        out = data.AccessAs<T>()[0];
    };

    switch(attributeIndex)
    {
        case  0: Load(newOptions.cacheEntryLimit, data); break;
        case  1: Load(newOptions.cachePosBits, data); break;
        case  2: Load(newOptions.cacheNormalBits, data); break;
        case  3: Load(newOptions.cacheLevelCount, data); break;
        case  4: Load(newOptions.cacheConeAperture, data); break;
        //
        case  5: Load(newOptions.russianRouletteRange, data); break;
        case  6: Load(newOptions.totalSPP, data); break;
        case  7: newOptions.lightSampler = LightSamplerType(std::as_const(data).AccessAsString()); break;
        case  8: newOptions.renderMode = RenderMode(std::as_const(data).AccessAsString()); break;
        case  9: Load(newOptions.burstSize, data); break;
        case 10: Load(newOptions.lobeProbablity, data); break;
        case 11: newOptions.displayMode = DisplayMode(std::as_const(data).AccessAsString()); break;
        default:
            throw MRayError("{} Unknown attribute index {}", TypeName(), attributeIndex);
    }
}

GuidedPTRDetail::RayState
GuidedPTRenderer::PackRayState() const
{
    RayState rs = RayState{};
    rs.dPathRadiance        = dPathRadiance;
    rs.dImageCoordinates    = dImageCoordinates;
    rs.dFilmFilterWeights   = dFilmFilterWeights;
    rs.dPathWavelengths     = dPathWavelengths;
    rs.dThroughputs         = dThroughputs;
    rs.dPathDataPack        = dPathDataPack;
    rs.dPrevPDF             = dPrevPDF;
    rs.dScoreSums           = dScoreSums;
    rs.dBackupRNGStates     = rnGenerator->GetBackupStates();
    rs.dShadowRayRadiance   = dShadowRayRadiance;
    rs.dShadowRays          = dShadowRays;
    rs.dShadowRayCones      = dShadowRayCones;
    rs.dLiftedMCIndices     = dLiftedMCIndices;
    //
    rs.dPrevPathReflectanceOrOutRadiance = dPrevPathReflectanceOrOutRadiance;
    rs.dShadowPrevPathOutRadiance = dShadowPrevPathOutRadiance;
    return rs;
}

GuidedPTRDetail::GlobalState
GuidedPTRenderer::PackGlobalState() const
{
    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    auto lightSampler = UniformLightSampler(metaLightArray.Array(),
                                            metaLightArray.IndexHashTable());
    return GlobalState
    {
        .russianRouletteRange = currentOptions.russianRouletteRange,
        .specContextData      = typedSpectrumContext.GetData(),
        .lightSampler         = lightSampler,
        .lobeProbability      = currentOptions.lobeProbablity,
        // Hash Grid and Data
        .hashGrid             = hashGrid.View(),
        .dMCStates            = dMCStates,
        .dMCCounts            = dMCCounts,
        .dMCLocks             = dMCLocks,
        .dMCIrradiances       = dMCIrradiances
    };
}

typename GuidedPTRenderer::ImageSectionOpt
GuidedPTRenderer::DisplayHashGrid(Span<const RayIndex> dDeadRayIndices,
                                  const GPUQueue& processQueue,
                                  const GPUQueue& transferQueue)
{
    if(dDeadRayIndices.size() == 0) return std::nullopt;

    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    // We lost the camera rays of these paths, rays are dead anyway.
    // Luckily we could just use the old buffers.
    curCamWork->ReconstructCameraRays(dShadowRayCones, dShadowRays,
                                      dImageCoordinates, dDeadRayIndices,
                                      dSubCameraBuffer, curCamTransformKey,
                                      imageTiler.CurrentTileSize(),
                                      processQueue);
    // Cast rays for tMax (we will not use primitives for simlicity and find pos via
    // ray hit position).
    tracerView.baseAccelerator.CastRays
    (
        dHitKeys, dHits, dBackupRNGStates,
        dShadowRays, dDeadRayIndices, processQueue
    );

    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    renderBuffer->ClearImage(processQueue);
    //
    ImageSpan filmSpan = imageTiler.GetTileSpan();
    processQueue.IssueWorkKernel<KCWriteHashGridDataToImg>
    (
        "KCWriteHashGridDataToImg",
        DeviceWorkIssueParams{.workCount = uint32_t(dDeadRayIndices.size())},
        //
        filmSpan,
        dBackupRNGStates,
        dShadowRays,
        dDeadRayIndices,
        dFilmFilterWeights,
        dImageCoordinates,
        hashGrid.View(),
        dMCStates,
        dMCCounts,
        dMCIrradiances,
        DisplayMode::E(currentOptions.displayMode)
    );

    // Issue a send of the FBO to Visor
    ImageSectionOpt renderOut = imageTiler.TransferToHost(processQueue,
                                                          transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value()) return std::nullopt;
    //
    renderOut->globalWeight = Float(1);
    return renderOut;
}

uint32_t
GuidedPTRenderer::FindMaxSamplePerIteration(uint32_t rayCount)
{
    uint32_t camSample = curCamWork->StochasticFilterSampleRayRNList().TotalRNCount();
    uint32_t spectrumSample = (spectrumContext)
                ? spectrumContext->SampleSpectrumRNList().TotalRNCount()
                : 0u;

    uint32_t maxSample = Math::Max(camSample, spectrumSample);
    maxSample = std::transform_reduce
    (
        currentWorks.cbegin(), currentWorks.cend(), maxSample,
        [](uint32_t l, uint32_t r) -> uint32_t
        {
            return Math::Max(l, r);
        },
        [](const auto& renderWorkStruct) -> uint32_t
        {
            return renderWorkStruct.workPtr->SampleRNList(0).TotalRNCount();
        }
    );
    return rayCount * maxSample;
}

Span<RayIndex>
GuidedPTRenderer::DoRenderPass(uint32_t sppLimit,
                               const GPUQueue& processQueue)
{
    assert(sppLimit != 0);

    // Create RNG state for each ray
    rnGenerator->SetupRange(imageTiler.LocalTileStart(),
                            imageTiler.LocalTileEnd(),
                            processQueue);
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    //
    GlobalState globalState = PackGlobalState();
    RayState dRayState = PackRayState();

    // Fill dead rays according to the render mode
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto [dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                  processQueue, true);
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);

    // Reload dead paths with new
    auto [dReloadIndices, dFilledIndices, aliveRayCount] = ReloadPaths(dIndices, sppLimit,
                                                                       processQueue);
    dIndices = dReloadIndices.subspan(0, aliveRayCount);
    dKeys = dKeys.subspan(0, aliveRayCount);

    // New rays do not have back path segment so set it
    if(!dFilledIndices.empty())
    {
        processQueue.IssueWorkKernel<KCInitGuidedPathStatesIndirect>
        (
            "KCInitGuidedPathStatesIndirect",
            DeviceWorkIssueParams{.workCount = uint32_t(dFilledIndices.size())},
            //
            dLiftedMCIndices,
            ToConstSpan(dFilledIndices)
        );
    }
    // From this point on we have full buffers,
    // unless we are about to reach the spp limit
    // them some rays are marked invalid.
    // ============ //
    // Ray Casting  //
    // ============ //
    processQueue.IssueWorkKernel<KCSetBoundaryWorkKeysIndirect>
    (
        "KCSetBoundaryWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        dHitKeys,
        ToConstSpan(dIndices),
        this->boundaryLightKeyPack
    );
    // Actual Ray Casting
    tracerView.baseAccelerator.CastRays
    (
        dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, processQueue
    );
    // Generate work keys from hit packs
    // for partitioning
    using namespace std::string_literals;
    processQueue.IssueWorkKernel<KCGenerateWorkKeysIndirect>
    (
        "KCGenerateWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        dKeys,
        ToConstSpan(dIndices),
        ToConstSpan(dHitKeys),
        workHasher
    );
    // Finally, partition using the generated keys.
    // Fully partitioning here by using a single sort
    auto& rp = rayPartitioner;
    auto partitionOutput = rp.MultiPartition(dKeys, dIndices,
                                             workHasher.WorkBatchDataRange(),
                                             workHasher.WorkBatchBitRange(),
                                             processQueue, false);
    // Wait for results to be available in host buffers
    // since we need partition ranges on the CPU to Issue kernels.
    processQueue.Barrier().Wait();
    // Old Indices array (and the key) is invalidated
    // due to how partitioner works.
    // Change indices to the partitioned one
    dIndices = partitionOutput.dPartitionIndices;

    // Before calcuating new path, backpropogate the irradiance
    // for non-light hitting rays.
    // We could've does this as a "Shader" but we can get away with
    // tMM as position, and ray direction as normal
    //
    // Clear locks
    // TODO: Memset or indirect update via kernel? (which has better perf?)
    processQueue.MemsetAsync(dMCLocks, 0x00);
    // Repurpose "shadowRays" buffer for MCStateLifting
    Span<TransientMCState> dLiftedMCStates = MemAlloc::RepurposeAlloc<TransientMCState>(dShadowRays);
    // Indirect irradiance backpropogation (as well as updater selection)
    processQueue.IssueWorkKernel<KCBackpropagateHashGridPath>
    (
        "KCBackpropagateHashGridPath",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        //
        dLiftedMCStates,
        globalState,
        dLiftedMCIndices,
        dBackupRNGStates,
        dPrevPathReflectanceOrOutRadiance,
        dScoreSums,
        dHitKeys,
        dRays,
        dIndices
    );
    processQueue.IssueWorkKernel<KCWriteMarkovChains>
    (
        "KCWriteMarkovChains",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        //
        globalState,
        dBackupRNGStates,
        dLiftedMCStates,
        dLiftedMCIndices,
        dIndices,
        dHitKeys
    );

    // Clear the shadow ray stuff
    processQueue.MemsetAsync(dShadowRayRadiance, 0x00);
    processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
    processQueue.MemsetAsync(dShadowRays, 0x00);
    processQueue.MemsetAsync(dShadowPrevPathOutRadiance, 0x00);
    //
    IssueWorkKernelsToPartitions<This>
    (
        workHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t partitionSize)
        {
            RNRequestList rnList = workI.SampleRNList(0);
            uint32_t rnCount = rnList.TotalRNCount();
            auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
            rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                 dPathRNGDimensions,
                                                 rnList,
                                                 processQueue);
            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, dLocalIndices, processQueue,
                ConstAddFunctor_U16(uint16_t(rnCount))
            );

            // 1. vMF Generation & Sampling / Shadow Ray Generation
            // 2. Virtually lifting a Markov Chain
            // 3. Path & Shadow ray single-channel irradiance
            //    throughput generation
            //
            workI.DoWork_0(dRayState, dRays, dRayCones,
                           dLocalIndices, dRandomNumBuffer,
                           dHits, dHitKeys,
                           globalState, processQueue);
        },
        // Light selection
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            // Light emission calculation / direct backpropogation
            // of irradiance
            workI.DoBoundaryWork_0(dRayState, dRays, dRayCones,
                                   dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dHits, dHitKeys,
                                   globalState, processQueue);
        }
    );

    // Check the shadow ray visibility
    Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
    tracerView.baseAccelerator.CastVisibilityRays
    (
        dIsVisibleBitSpan, dBackupRNGStates,
        dShadowRays, dIndices, processQueue
    );

    // Accumulate the pre-calculated radiance selectively
    processQueue.IssueWorkKernel<KCAccumulateShadowRaysGPT>
    (
        "KCAccumulateShadowRays",
        DeviceWorkIssueParams{.workCount = uint32_t(dShadowRayRadiance.size())},
        //
        dPathRadiance,
        ToConstSpan(dShadowRayRadiance),
        ToConstSpan(dIsVisibleBitSpan),
        ToConstSpan(dPathDataPack)
    );

    // Shadow Ray irradiance backpropogation
    processQueue.IssueWorkKernel<KCBackpropagateIrradNEE>
    (
        "KCBackpropagateIrradNEE",
        DeviceWorkIssueParams{.workCount = uint32_t(dLiftedMCIndices.size())},
        //
        globalState,
        ToConstSpan(dShadowPrevPathOutRadiance),
        ToConstSpan(dLiftedMCIndices),
        ToConstSpan(dPathDataPack),
        ToConstSpan(dIsVisibleBitSpan)
    );

    // Backprop light
    processQueue.IssueWorkKernel<KCBackpropagateIrradLight>
    (
        "KCBackpropagateIrradLight",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        //
        globalState,
        ToConstSpan(dLiftedMCIndices),
        ToConstSpan(dPrevPathReflectanceOrOutRadiance),
        ToConstSpan(dHitKeys),
        ToConstSpan(dIndices)
    );

    // Debug report the hash grid
    // TODO: Delete this later
    if(totalIterationCount % 64 == 0)
    {
        uint32_t count = hashGrid.UsedEntryCount(processQueue);
        MRAY_LOG("HashGrid Size: {}", count);
    }
    return dIndices;
}

RendererOutput
GuidedPTRenderer::DoThroughputSingleTileRender(const GPUDevice& device,
                                               const GPUQueue& processQueue)
{
    Timer timer; timer.Start();
    const auto& cameraWork = *curCamWork;
    // Generate subcamera of this specific tile
    if(totalIterationCount == 0)
    {
        cameraWork.GenerateSubCamera
        (
            dSubCameraBuffer,
            curCamKey, curCamTransformOverride,
            imageTiler.CurrentTileIndex(),
            imageTiler.TileCount(),
            processQueue
        );

        // TODO: We need a proper camera position acquisiton
        // system sometime later. (It can be useful like scene AABB)
        Vector3 camPos;
        cameraWork.GenCameraPosition(Span<Vector3, 1>(dCamPosBuffer),
                                     dSubCameraBuffer,
                                     curCamTransformKey,
                                     processQueue);
        processQueue.MemcpyAsync(Span<Vector3>(&camPos, 1), ToConstSpan(dCamPosBuffer));
        processQueue.MemsetAsync(dMCCounts, 0x00);
        processQueue.MemsetAsync(dMCStates, 0x00);
        processQueue.MemsetAsync(dMCIrradiances, 0x00);
        hashGrid.ClearAllEntries(processQueue);

        processQueue.Barrier().Wait();
        hashGrid.SetCameraPos(camPos);
    }

    // ====================== //
    //   Single Render Pass   //
    // ====================== //
    uint32_t sppLimit = (saveImage) ? currentOptions.totalSPP
                                    : std::numeric_limits<uint32_t>::max();
    Span<RayIndex> dIndices = DoRenderPass(sppLimit, processQueue);

    // Find the dead paths again
    // Do a 3-way partition here to catch potential invalid rays
    auto deadAlivePartitionOut = rayPartitioner.TernaryPartition
    (
        dIndices, processQueue,
        IsDeadAliveInvalidFunctor(ToConstSpan(dPathDataPack))
    );
    processQueue.Barrier().Wait();
    auto [dDeadRayIndices, dInvalidRayIndices, dAliveRayIndices] =
        deadAlivePartitionOut.Spanify();

    // Write radiance of dead rays to image buffer async
    // via transfer queue.
    Optional<RenderImageSection> renderOut;
    const GPUQueue& transferQueue = device.GetTransferQueue();
    if(currentOptions.displayMode == DisplayMode::E::RENDER)
    {
        renderOut = AddRadianceToRenderBufferThroughput(dDeadRayIndices,
                                                        processQueue,
                                                        transferQueue);
    }
    else
    {
        renderOut = DisplayHashGrid(dDeadRayIndices, processQueue,
                                    transferQueue);
    }

    // Copy continuing set of rays to actual ray buffer for next iteration
    //CopyAliveRays(dAliveRayIndices, processQueue);

    // We exhausted all alive rays while doing SPP limit.
    bool triggerSave = (saveImage && tilePathCounts[0] == TotalSampleLimit(currentOptions.totalSPP));
    if(triggerSave) saveImage = false;

    // We do not need to wait here, but we time
    // from CPU side so we need to wait
    // TODO: In future we should do OpenGL, Vulkan
    // style performance counters events etc. to
    // query the timing (may be couple of frame before even)
    // The timing is just a general performance indicator
    // It should not be super accurate.
    processQueue.Barrier().Wait();
    timer.Split();

    // Report the results
    totalIterationCount++;
    RendererAnalyticData analyticData;
    analyticData = CalculateAnalyticDataThroughput(dDeadRayIndices.size(),
                                                   currentOptions.totalSPP,
                                                   timer);
    analyticData.customLogicSize0 = uint32_t(DisplayMode::E::END);
    return RendererOutput
    {
        .analytics = std::move(analyticData),
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

RendererOutput
GuidedPTRenderer::DoLatencyRender(uint32_t passCount,
                                  const GPUDevice& device,
                                  const GPUQueue& processQueue)
{
    Vector2ui tileCount2D = imageTiler.TileCount();
    uint32_t tileCount1D = tileCount2D.Multiply();

    Timer timer; timer.Start();
    // Generate subcamera of this specific tile
    const auto& cameraWork = *curCamWork;
    cameraWork.GenerateSubCamera
    (
        dSubCameraBuffer,
        curCamKey, curCamTransformOverride,
        imageTiler.CurrentTileIndex(),
        tileCount2D,
        processQueue
    );
    // New camera so reset hash grid
    if(totalIterationCount == 0)
    {
        // TODO: We need a proper camera position acquisiton
        // system sometime later. (It can be useful like scene AABB)
        Vector3 camPos;
        cameraWork.GenCameraPosition(Span<Vector3, 1>(dCamPosBuffer),
                                     dSubCameraBuffer,
                                     curCamTransformKey,
                                     processQueue);
        processQueue.MemcpyAsync(Span<Vector3>(&camPos, 1), ToConstSpan(dCamPosBuffer));
        processQueue.MemsetAsync(dMCCounts, 0x00);
        processQueue.MemsetAsync(dMCStates, 0x00);
        processQueue.MemsetAsync(dMCIrradiances, 0x00);
        hashGrid.ClearAllEntries(processQueue);

        processQueue.Barrier().Wait();
        hashGrid.SetCameraPos(camPos);
    }

    // We are waiting too early here,
    // We should wait at least on the first render buffer write
    // but it was not working so I've put it here
    // TODO: Investigate
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    renderBuffer->ClearImage(processQueue);

    //
    uint32_t tileIndex = imageTiler.CurrentTileIndex1D();
    uint32_t tileSPP = static_cast<uint32_t>(tileSPPs[tileIndex]);
    uint32_t sppLimit = tileSPP + passCount;
    if(saveImage)
        sppLimit = Math::Min(sppLimit, currentOptions.totalSPP);

    tileSPPs[tileIndex] = sppLimit;
    uint32_t currentPassCount = sppLimit - tileSPP;
    uint32_t passPathCount = imageTiler.CurrentTileSize().Multiply() * currentPassCount;
    uint32_t invalidRayCount = 0;
    do
    {
        Span<RayIndex> dIndices = DoRenderPass(sppLimit, processQueue);

        // Find the dead paths again
        // Every path is processed, so we do not need to use the scrambled
        // index buffer. Iota again
        //DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
        // Do a 3-way partition,
        auto deadAlivePartitionOut = rayPartitioner.TernaryPartition
        (
            dIndices, processQueue,
            IsDeadAliveInvalidFunctor(ToConstSpan(dPathDataPack))
        );
        processQueue.Barrier().Wait();
        auto [dDeadRayIndices, dInvalidRayIndices, dAliveRayIndices] =
            deadAlivePartitionOut.Spanify();

        AddRadianceToRenderBufferLatency(dDeadRayIndices, processQueue);

        assert(dInvalidRayIndices.size() == 0);
        invalidRayCount += (static_cast<uint32_t>(dDeadRayIndices.size()));
    } while(invalidRayCount != passPathCount);

    // One spp of this tile should be done now.
    // Issue the transfer
    // Issue a send of the FBO to Visor
    const GPUQueue& transferQueue = device.GetTransferQueue();
    Optional<RenderImageSection> renderOut;
    renderOut = imageTiler.TransferToHost(processQueue, transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value()) return RendererOutput{};
    // Actual global weight
    renderOut->globalWeight = Float(1);

    // We do not need to wait here, but we time
    // from CPU side so we need to wait
    // TODO: In future we should do OpenGL, Vulkan
    // style performance counters events etc. to
    // query the timing (may be couple of frame before even)
    // The timing is just a general performance indicator
    // It should not be super accurate.
    processQueue.Barrier().Wait();
    timer.Split();

    // Roll to the next tile
    imageTiler.NextTile();

    // Check save trigger
    // Notice, we've changed to the next tile above.
    uint64_t curSPPSum = std::reduce(tileSPPs.cbegin(), tileSPPs.cend(), uint64_t(0));
    uint64_t sppSumCheck = (uint64_t(currentOptions.totalSPP) *
                            uint64_t(tileCount1D));
    bool triggerSave = (saveImage && curSPPSum == sppSumCheck);
    if(triggerSave) saveImage = false;

    // Report the results
    totalIterationCount++;
    RendererAnalyticData analyticData;
    analyticData = CalculateAnalyticDataLatency(passPathCount, currentOptions.totalSPP,
                                                timer);
    analyticData.customLogicSize0 = uint32_t(DisplayMode::E::END);

    return RendererOutput
    {
        .analytics   = std::move(analyticData),
        .imageOut    = renderOut,
        .triggerSave = triggerSave
    };
}

RenderBufferInfo
GuidedPTRenderer::StartRender(const RenderImageParams& rIP,
                              CamSurfaceId camSurfId,
                              uint32_t customLogicIndex0,
                              uint32_t)
{
    currentOptions = newOptions;
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);

    // Change the mode according to the render logic
    using Math::Roll;
    int32_t modeIndex = (int32_t(DisplayMode::E(currentOptions.displayMode)) +
                         int32_t(customLogicIndex0));
    uint32_t sendMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                      int32_t(DisplayMode::E::END)));
    uint32_t newMode = uint32_t(Roll(modeIndex, 0, int32_t(DisplayMode::E::END)));
    currentOptions.displayMode = DisplayMode::E(newMode);
    MRAY_LOG("NewMode! {}", newMode);

    // ================================ //
    // Initialize common sub components //
    // ================================ //
    uint32_t sppLimit = currentOptions.totalSPP;
    auto [maxRayCount, totalWorkCount] = InitializeForRender(camSurfId, sppLimit,
                                                             false, rIP);
    renderMode = currentOptions.renderMode;
    burstSize = currentOptions.burstSize;

    // ========================= //
    //      Spectrum Context     //
    // ========================= //
    uint32_t wavelengthCount = (IsSpectral) ? maxRayCount : 0u;
    auto colorSpace = tracerView.tracerParams.globalTextureColorSpace;
    if constexpr(IsSpectral)
    {
        // Don't bother reloading context if colorspace is same
        if(!spectrumContext || colorSpace != spectrumContext->ColorSpace())
        {
            using SC = SpectrumContext;
            auto wlSampleMode = tracerView.tracerParams.wavelengthSampleMode;
            spectrumContext = std::make_unique<SC>(colorSpace, wlSampleMode,
                                                   gpuSystem);
        }
    }

    // ========================= //
    //         Hash Grid         //
    // ========================= //
    hashGrid.Reset(tracerView.baseAccelerator.SceneAABB(),
                   // We do not know te cam pos yet,
                   Vector3::Zero(),
                   currentOptions.cachePosBits,
                   currentOptions.cacheNormalBits,
                   currentOptions.cacheLevelCount,
                   currentOptions.cacheConeAperture,
                   currentOptions.cacheEntryLimit,
                   queue);

    // ========================= //
    //   Path State Allocation   //
    // ========================= //
    // You can see why wavefront approach uses
    // quite a bit memory (and this is somewhat optimized).
    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount);
    uint32_t hashGridEntryCount = hashGrid.EntryCapacity();
    uint32_t isVisibleIntCount = Bitspan<uint32_t>::CountT(maxRayCount);
    MemAlloc::AllocateMultiData
    (
        Tie
        (
            // Per path
            dHits, dHitKeys, dRays, dRayCones,
            dPathRadiance, dImageCoordinates,
            dFilmFilterWeights, dThroughputs,
            dPathDataPack, dPrevPDF,
            dShadowPrevPathOutRadiance,
            dPrevPathReflectanceOrOutRadiance,
            dScoreSums, dLiftedMCIndices,
            dShadowRays, dShadowRayCones,
            dShadowRayRadiance, dPathRNGDimensions,
            // Per path (but can be zero if not spectral)
            dPathWavelengths,
            dSpectrumWavePDFs,
            // Per path but a single bit is used
            dShadowRayVisibilities,
            // Per path bounce, max used random number of that bounce
            dRandomNumBuffer,
            // Per render work
            dWorkHashes, dWorkBatchIds,
            // Hash Grid
            dMCStates, dMCLocks,
            dMCCounts, dMCIrradiances,
            //
            dSubCameraBuffer,
            dCamPosBuffer
        ),
        rendererGlobalMem,
        {
            // Per path
            maxRayCount, maxRayCount, maxRayCount, maxRayCount,
            maxRayCount, maxRayCount, maxRayCount, maxRayCount,
            maxRayCount, maxRayCount, maxRayCount, maxRayCount,
            maxRayCount, maxRayCount, maxRayCount, maxRayCount,
            maxRayCount, maxRayCount,
            // Per path (but can be zero if not spectral)
            wavelengthCount, wavelengthCount,
            // Per path but a single bit is used
            isVisibleIntCount,
            // Per path bounce, max used random number of that bounce
            maxSampleCount,
            // Per render work
            totalWorkCount, totalWorkCount,
            // Hash Grid
            hashGridEntryCount, hashGridEntryCount,
            hashGridEntryCount, hashGridEntryCount,
            //
            RendererBase::SUB_CAMERA_BUFFER_SIZE,
            size_t(1)
        }
    );
    MetaLightListConstructionParams mlParams =
    {
        .lightGroups = tracerView.lightGroups,
        .transformGroups = tracerView.transGroups,
        .lSurfList = Span<const Pair<LightSurfaceId, LightSurfaceParams>>(tracerView.lightSurfs)
    };
    metaLightArray.Construct(mlParams, tracerView.boundarySurface, queue);

    // ===================== //
    // After Allocation Init //
    // ===================== //
    workHasher = InitializeHashes(dWorkHashes, dWorkBatchIds,
                                  maxRayCount, queue);

    // Reset hits and paths
    queue.MemsetAsync(dHits, 0x00);
    ResetAllPaths(queue);

    auto bufferPtrAndSize = renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = colorSpace,
        .resolution = imageTiler.FullResolution(),
        .curRenderLogic0 = sendMode,
        .curRenderLogic1 = std::numeric_limits<uint32_t>::max()
    };
}

void GuidedPTRenderer::StopRender()
{
    ClearAllWorkMappings();
    filmFilter = {};
    rnGenerator = {};
    metaLightArray.Clear();
}

std::string_view
GuidedPTRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "GuidedPathTracerSpectral"sv;
    return RendererTypeName<Name>;
}

typename GuidedPTRenderer::AttribInfoList
GuidedPTRenderer::StaticAttributeInfo()
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;

    return AttribInfoList
    {
        {"cacheEntryLimit",   MRayDataTypeRT(MR_UINT32),     IS_SCALAR, MR_OPTIONAL },
        {"cachePosBits",      MRayDataTypeRT(MR_UINT32),     IS_SCALAR, MR_OPTIONAL },
        {"cacheNormalBits",   MRayDataTypeRT(MR_UINT32),     IS_SCALAR, MR_OPTIONAL },
        {"cacheLevelCount",   MRayDataTypeRT(MR_UINT32),     IS_SCALAR, MR_OPTIONAL },
        {"cacheConeAperture", MRayDataTypeRT(MR_FLOAT),      IS_SCALAR, MR_OPTIONAL },
        {"rrRange",           MRayDataTypeRT(MR_VECTOR_2UI), IS_SCALAR, MR_MANDATORY},
        {"totalSPP",          MRayDataTypeRT(MR_UINT32),     IS_SCALAR, MR_MANDATORY},
        {"lightSampler",      MRayDataTypeRT(MR_STRING),     IS_SCALAR, MR_OPTIONAL },
        {"renderMode",        MRayDataTypeRT(MR_STRING),     IS_SCALAR, MR_MANDATORY},
        {"burstSize",         MRayDataTypeRT(MR_UINT32),     IS_SCALAR, MR_OPTIONAL },
        {"lobeProbability",   MRayDataTypeRT(MR_FLOAT),      IS_SCALAR, MR_OPTIONAL },
        {"displayMode",       MRayDataTypeRT(MR_STRING),     IS_SCALAR, MR_MANDATORY},
    };
}

size_t GuidedPTRenderer::GPUMemoryUsage() const
{
    size_t total = (rayPartitioner.GPUMemoryUsage() +
                    rnGenerator->GPUMemoryUsage() +
                    rendererGlobalMem.Size());
    if(spectrumContext)
        total += spectrumContext->GPUMemoryUsage();

    return total;
}