#pragma once

#include "Core/NamedEnum.h"
#include "Core/Definitions.h"
#include "Core/Vector.h"

#include "Tracer/PathTracerRendererBase.h"
#include "Tracer/HashGrid.h"
#include "Tracer/SpectrumContext.h"
#include "Tracer/PrimitiveC.h"
#include "Tracer/MaterialC.h"
#include "Tracer/RenderWork.h"
#include "Tracer/LightSampler.h"

#include "RequestedTypes.h"

namespace GuidedPTRDetail
{
    // Slightly compressed Markov Chain data of
    // Alber et al. (2025).
    using MCTarget     = Vector3;
    using MCSumAndCos  = Vector2;
    using MCIrradiance = Float;
    //
    struct alignas(4) MCCount
    {
        uint16_t mcN;
        uint16_t irradN;
    };

    enum class DisplayModeEnum
    {
        RENDER,
        LIGHT_CACHE,
        //
        END
    };
    inline constexpr std::array DisplayModeNames =
    {
        "Render",
        "LightCache"
    };
    using DisplayMode = NamedEnum<DisplayModeEnum, DisplayModeNames>;

    struct Options
    {
        // ====================== //
        //    Hash Grid Related   //
        // ====================== //
        uint32_t cacheEntryLimit    = 2'000'000; // At most 2M entries
                                                 // (allocation is 2x 2M = nearest pow2)
        uint32_t cachePosBits       = 16;        // Maximum subdiv of bottom level voxel
                                                 // SceneAABB / 2^16 = voxel size
        uint32_t cacheNormalBits    = 2;         // Concentric Octrahedral map of the normal
                                                 // which will be divided on to 4x4 grid
        uint32_t cacheLevelCount    = 8;         // Maximum upper levels of the voxel grid
        Float    cacheConeAperture = Float(0.6); // When a ray hits a cache entry, it will be assumed
                                                 // has a differential as if it had a ray cone with this
                                                 // aperture.
        // ====================== //
        //  Path Tracing Related  //
        // ====================== //
        // Classic RR. No RR until 2 bounce,
        // at most 20 bounces may occur (aka. depth limit).
        Vector2ui        russianRouletteRange = Vector2ui(2, 20);
        // Total samples that will be traced per pixel.
        uint32_t         totalSPP     = 16'384;
        // NEE Sampler
        LightSamplerType lightSampler = LightSamplerType::E::UNIFORM;
        //
        // Mode of the renderer
        //
        // Throughput: Single buffer of paths
        // will be continuously churned.
        // If a ray dies (RR, hits light) a new
        // ray from a different pixel will have its place
        // system will stop adding paths when it reaches to the totalSPP.
        //
        // Throughput mode is only available when image is not large and tiling is not triggered
        // this mode should be the fastest mode but triggers flickering when moving the camera.
        //
        // Latency: N_SPP amount of paths will be generated. These will be traced
        // until all paths are terminated.
        RenderMode       renderMode = RenderMode::E::THROUGHPUT;
        // N parameter of the latency mode
        uint32_t         burstSize = 1;

        // ================== //
        // Alber2025 Related  //
        // ================== //
        // Limit probably of lobe selection.
        // At start this will be lower, this is the maximum
        // on the equilibrium.
        Float       lobeProbablity = Float(0.7);
        // For debugging etc.
        DisplayMode displayMode;
    };

    using LightSampler = DirectLightSamplerUniform<MetaLightList>;

    struct GlobalState
    {
        using SpecConverterData = typename SpectrumContextJakob2019::Data;

        Vector2ui           russianRouletteRange;
        Vector3             guideBxDFNEEProbablity;
        SpecConverterData   specContextData;
        LightSampler        lightSampler;
        Float               lobeProbability;
        // Hash Grid and Alber2025 data
        HashGridView        hashGrid;
        Span<MCTarget>      dMarkovWTargets;
        Span<MCSumAndCos>   dMarkovWCosAndWSums;
        Span<MCCount>       dMarkovCounts;
        Span<MCIrradiance>  dMarkovIrradParams;
    };

    struct RayState
    {
        Span<Spectrum>          dPathRadiance;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
        Span<SpectrumWaves>     dPathWavelengths;
        // Path state
        Span<Spectrum>          dThroughput;
        Span<PathDataPack>      dPathDataPack;
        Span<Float>             dPDFChain;
        // Next set of rays
        Span<RayGMem>           dOutRays;
        Span<RayCone>           dOutRayCones;
        // Backup RNG State
        Span<BackupRNGState>    dBackupRNGStates;
        // NEE Related
        Span<Float>             dPrevMatPDF;
        Span<Spectrum>          dShadowRayRadiance;
        // Current lifted markov chain
        Span<SpatioDirCode>     dPrevHitCodes;
        Span<uint32_t>          dLiftedMarkovChainIndex;
    };

    using GaussianLobe = Distribution::GaussianLobe;

    template<uint32_t N>
    struct GaussLobeMixtureT
    {
        static constexpr uint32_t LobeCount     = N;
        static constexpr uint32_t SampleRNCount = 2;

        std::array<GaussianLobe, N> lobes;
        std::array<Float, N>        weights;
        uint32_t                    sampleIndex = UINT32_MAX;
        Float                       sumWeight   = 0;
        // Functionality
        MR_GF_DECL SampleT<Vector3> Sample(RNGDispenser& rng) const;
        MR_GF_DECL Float            Pdf(const Vector3& wO) const;
        MR_GF_DECL uint32_t         LoadStochastic(const Vector3& position,
                                                   const Vector3& normal,
                                                   //
                                                   BackupRNG& rng,
                                                   //
                                                   const GlobalState& gs);
    };

    template<uint32_t N>
    struct GaussLobeMixtureSharedT
    {
        static constexpr uint32_t LobeCount     = N;
        static constexpr uint32_t SampleRNCount = 2;
        //
        struct Storage
        {
            Float sLobeX[StaticThreadPerBlock1D() * N];
            Float sLobeY[StaticThreadPerBlock1D() * N];
            Float sLobeZ[StaticThreadPerBlock1D() * N];
            Float sLobeKappa[StaticThreadPerBlock1D() * N];
            Float sLobeAlpha[StaticThreadPerBlock1D() * N];
            Float sLobeWeights[StaticThreadPerBlock1D() * N];
        };
        // Thread-local data
        uint32_t                    sampleIndex = UINT32_MAX;
        Float                       sumWeight   = 0;
        // Shared memory data
        MR_GF_DECL Storage&         SMem() const;
        MR_GF_DECL static uint32_t  Index(uint32_t i);
        // Functionality
        MR_GF_DECL SampleT<Vector3> Sample(RNGDispenser& rng) const;
        MR_GF_DECL Float            Pdf(const Vector3& wO) const;
        MR_GF_DECL uint32_t         LoadStochastic(const Vector3& position,
                                                   const Vector3& normal,
                                                   //
                                                   BackupRNG& rng,
                                                   //
                                                   const GlobalState& gs);
    };

    static constexpr uint32_t MC_LOBE_COUNT = 8;

    // TODO: Macro for CPU/GPU after profiling
    //using GaussianLobeMixture = GaussLobeMixtureSharedT<MC_LOBE_COUNT>;
    using GaussianLobeMixture = GaussLobeMixtureT<MC_LOBE_COUNT>;

    MR_HF_DECL
    GaussianLobe SufficientStatsToLobe(const Vector3& wTarget,
                                       Float wSum, Float wCos,
                                       uint32_t mcSampleCount,
                                       const Vector3& samplePos);

    template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkParams = RenderWorkParams<GlobalState, RayState, PG, MG, TG>;

    template<LightGroupC LG, TransformGroupC TG>
    using LightWorkParams = RenderLightWorkParams<GlobalState, RayState, LG, TG>;

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType>
    struct WorkFunction
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType,
                                       SpectrumContextJakob2019, 1u);
        using Params = WorkParams<PG, MG, TG>;

        MR_GF_DECL
        static void Call(const Primitive&, const Material&, const Surface&,
                         const RayConeSurface&, const TContext&,
                         SpectrumConv&, RNGDispenser&,
                         const Params& params,
                         RayIndex rayIndex, uint32_t laneId);
    };

    template<LightGroupC LGType, TransformGroupC TGType>
    struct LightWorkFunction
    {
        MRAY_LIGHT_WORK_FUNCTOR_DEFINE_TYPES(LGType, TGType,
                                             SpectrumContextJakob2019, 1u);
        using Params = LightWorkParams<LG, TG>;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params& params, RayIndex rayIndex, uint32_t laneId);

    };

}

namespace GuidedPTRDetail
{

MR_HF_DEF
GaussianLobe SufficientStatsToLobe(const Vector3& wTarget,
                                   Float wSum,
                                   Float wCos,
                                   uint32_t mcSampleCount,
                                   const Vector3& samplePos)
{
    static constexpr Float DirGuidePrior = Float(0.2);
    static constexpr Float Epsilon       = Float(0.0001);
    static constexpr Float MaxCos        = Float(1) - Epsilon;

    // Fetch direction
    Vector3 target = wTarget;
    target = (wSum > 0) ? (target / wSum) : target;

    // Dir
    Vector3 dir = target - samplePos;
    Float distSqr = Math::LengthSqr(dir);
    Float priorState = Math::Max(Epsilon, DirGuidePrior / distSqr);
    dir /= Math::Sqrt(distSqr);

    // Kappa
    Float nSqr = Float(mcSampleCount) * Float(mcSampleCount);
    Float meanCos = nSqr * Math::Clamp(wCos / wSum, Float(0), MaxCos);
    meanCos = nSqr / priorState;
    // Mean cosine to kappa
    Float r = meanCos;
    Float r2 = r * r;
    Float r3 = r * r2;
    Float kappa = (Float(3) * r - r3) / (Float(1) - r2);
    return GaussianLobe(dir, kappa);
}

template<uint32_t N>
MR_GF_DEF
SampleT<Vector3>
GaussLobeMixtureT<N>::Sample(RNGDispenser& rng) const
{
    if(weights[sampleIndex] == Float(0))
    {
        return SampleT<Vector3>
        {
            .pdf = Float(0)
        };
    }
    auto sample = lobes[sampleIndex].Sample(rng.NextFloat2D<0>());
    return SampleT<Vector3>
    {
        .value = sample.value,
        .pdf = Pdf(sample.value)
    };
}

template<uint32_t N>
MR_GF_DEF
Float GaussLobeMixtureT<N>::Pdf(const Vector3& wO) const
{
    Float pdf = Float(0);
    for(uint32_t i = 0; i < N; i++)
        pdf += weights[i] * lobes[i].Pdf(wO);
    return pdf / sumWeight;
}

template<uint32_t N>
MR_GF_DEF
uint32_t GaussLobeMixtureT<N>::LoadStochastic(const Vector3& position,
                                              const Vector3& normal,
                                              //
                                              BackupRNG& rng,
                                              //
                                              const GlobalState& gs)
{
    static constexpr auto N = GaussianLobeMixture::LobeCount;
    uint32_t liftedNodeIndex = UINT32_MAX;
    for(uint32_t i = 0; i < N; i++)
    {
        // TODO: Sample with randomness
        Vector3 posJitter = position + Vector3(rng.NextFloat());
        Vector3 nJitter   = Math::Normalize(normal + Vector3(rng.NextFloat()));
        auto [hashGridIndex, _] = gs.hashGrid.TryInsertAtomic(posJitter, nJitter);

        Vector3 wTarget = gs.dMarkovWTargets[hashGridIndex];
        Float wSum      = gs.dMarkovWCosAndWSums[hashGridIndex][0];
        Float wCos      = gs.dMarkovWCosAndWSums[hashGridIndex][1];
        uint32_t mcN    = gs.dMarkovCounts[hashGridIndex].mcN;
        lobes[i]        = SufficientStatsToLobe(wTarget, wSum, wCos, mcN, position);
        weights[i]      = wSum;
        sumWeight      += wSum;
        //
        Float sumRatio = (sumWeight != Float(0)) ? (wSum / sumWeight) : Float(1);
        if(rng.NextFloat() < sumRatio)
        {
            sampleIndex = i;
            liftedNodeIndex = hashGridIndex;
        }
    }
    return liftedNodeIndex;
}

template<uint32_t N>
MR_GF_DEF
uint32_t GaussLobeMixtureSharedT<N>::Index(uint32_t i)
{
    auto tid = KernelCallParams().threadId;
    return i * N + tid;
}

template<uint32_t N>
MR_GF_DEF
typename GaussLobeMixtureSharedT<N>::Storage&
GaussLobeMixtureSharedT<N>::SMem() const
{
    extern MRAY_SHARED_MEMORY Byte s[];
    return *reinterpret_cast<Storage*>(s);
}

template<uint32_t N>
MR_GF_DEF
SampleT<Vector3>
GaussLobeMixtureSharedT<N>::Sample(RNGDispenser& rng) const
{
    if(SMem().sLobeWeights[Index(sampleIndex)] == Float(0))
    {
        return SampleT<Vector3>
        {
            .pdf = Float(0)
        };
    }
    GaussianLobe l(Vector3(SMem().sLobeX[Index(sampleIndex)],
                           SMem().sLobeY[Index(sampleIndex)],
                           SMem().sLobeZ[Index(sampleIndex)]),
                   SMem().sLobeKappa[Index(sampleIndex)],
                   SMem().sLobeAlpha[Index(sampleIndex)]);
    auto sample = l.Sample(rng.NextFloat2D<0>());
    return SampleT<Vector3>
    {
        .value = sample.value,
        .pdf = Pdf(sample.value)
    };
}

template<uint32_t N>
MR_GF_DEF
Float GaussLobeMixtureSharedT<N>::Pdf(const Vector3& wO) const
{
    Float pdf = Float(0);
    for(uint32_t i = 0; i < N; i++)
    {
        GaussianLobe l(Vector3(SMem().sLobeX[Index(i)],
                               SMem().sLobeY[Index(i)],
                               SMem().sLobeZ[Index(i)]),
                       SMem().sLobeKappa[Index(i)],
                       SMem().sLobeAlpha[Index(i)]);
        pdf += SMem().sLobeWeights[Index(i)] * l.Pdf(wO);
    }
    return pdf / sumWeight;
}

template<uint32_t N>
MR_GF_DEF
uint32_t GaussLobeMixtureSharedT<N>::LoadStochastic(const Vector3& position,
                                                    const Vector3& normal,
                                                    //
                                                    BackupRNG& rng,
                                                    //
                                                    const GlobalState& gs)
{
    static constexpr auto N = GaussianLobeMixture::LobeCount;
    uint32_t liftedNodeIndex = UINT32_MAX;
    for(uint32_t i = 0; i < N; i++)
    {
        // TODO: Sample with randomness
        Vector3 posJitter = position + Vector3(rng.NextFloat());
        Vector3 nJitter   = Math::Normalize(normal + Vector3(rng.NextFloat()));
        auto [hashGridIndex, _] = gs.hashGrid.TryInsertAtomic(posJitter, nJitter);

        Vector3 wTarget = gs.dMarkovWTargets[hashGridIndex];
        Float wSum      = gs.dMarkovWCosAndWSums[hashGridIndex][0];
        Float wCos      = gs.dMarkovWCosAndWSums[hashGridIndex][1];
        uint32_t mcN    = gs.dMarkovCounts[hashGridIndex].mcN;
        sumWeight      += wSum;

        auto lobe = SufficientStatsToLobe(wTarget, wSum, wCos, mcN, position);
        SMem().sLobeX[Index(i)] = lobe.dir[0];
        SMem().sLobeY[Index(i)] = lobe.dir[1];
        SMem().sLobeZ[Index(i)] = lobe.dir[2];
        SMem().sLobeKappa[Index(i)] = lobe.kappa;
        SMem().sLobeAlpha[Index(i)] = lobe.alpha;
        SMem().sLobeWeights[Index(i)] = wSum;

        Float sumRatio = (sumWeight != Float(0)) ? (wSum / sumWeight) : Float(1);
        if(rng.NextFloat() < sumRatio)
        {
            sampleIndex = i;
            liftedNodeIndex = hashGridIndex;
        }
    }
    return liftedNodeIndex;
}

// ================== //
//        PATH        //
// ================== //
template<PrimitiveGroupC P, MaterialGroupC M, TransformGroupC T>
MR_GF_DEF
void WorkFunction<P, M, T>::Call(const Primitive&, const Material& mat, const Surface& surf,
                                    const RayConeSurface& surfRayCone, const TContext& tContext,
                                    SpectrumConv& spectrumConverter, RNGDispenser& rng, const Params& params,
                                    RayIndex rayIndex, uint32_t laneId)
{
    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    using Distribution::MIS::BalanceCancelled;
    static constexpr auto MISSampleOffset = Math::Max(GaussianLobeMixture::SampleRNCount,
                                                      Material::SampleRNList.TotalRNCount());
    const GlobalState& globalState = params.globalState;
    const RayState& rayState       = params.rayState;
    BackupRNG backupRNG   = BackupRNG(rayState.dBackupRNGStates[rayIndex]);
    PathDataPack dataPack = params.rayState.dPathDataPack[rayIndex];

    if(dataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    auto [rayIn, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    Vector3 wOWorld = Math::Normalize(-rayIn.dir);
    Vector3 wO = Math::Normalize(tContext.InvApplyN(wOWorld));
    RayConeSurface rConeRefract = mat.RefractRayCone(surfRayCone, wO);
    Float specularity = mat.Specularity();

    // We do two sample mis
    // NEE / MIS of Lobes and BxDF
    // ====================== //
    //          NEE           //
    // ====================== //
    const LightSampler& lightSampler = params.globalState.lightSampler;
    LightSample lightSample = lightSampler.SampleLight(rng, spectrumConverter,
                                                       surf.position,
                                                       surf.geoNormal,
                                                       rConeRefract);
    auto [shadowRay, shadowTMM] = lightSample.value.SampledRay(surf.position);

    // ====================== //
    // Sample Material or vMF //
    // ====================== //
    GaussianLobeMixture mixture;
    uint32_t liftedLobeIndex = mixture.LoadStochastic(surf.position,
                                                      surf.geoNormal,
                                                      backupRNG,
                                                      params.globalState);
    //
    std::array<Float, 2> misPDFs = {};
    std::array<Float, 2> misWeights = {};
    if(!MaterialCommon::IsSpecular(specularity))
    {
        misWeights[0] = Math::Lerp(globalState.lobeProbability,
                                   Float(1), specularity);
    }
    misWeights[1] = Float(1) - misWeights[0];

    BxDFSample combinedSample;
    if(rng.NextFloat<MISSampleOffset>() < misWeights[0])
    {
        // Sample Mixture
        auto mixtureSample = mixture.Sample(rng);
        Vector3 wILocal = Math::Normalize(tContext.InvApplyN(mixtureSample.value));
        combinedSample.wI = Ray(mixtureSample.value, surf.position);
        combinedSample.eval = mat.Evaluate(Ray(wILocal, surf.position), wO);
        misPDFs[0] = mixtureSample.pdf;
        misPDFs[1] = mat.Pdf(Ray(wILocal, surf.position), wO);
    }
    else
    {
        // Sample Material
        combinedSample = mat.SampleBxDF(wO, rng);
        combinedSample.wI = tContext.Apply(combinedSample.wI);
        misPDFs[0] = mixture.Pdf(combinedSample.wI.dir);
        misPDFs[1] = combinedSample.pdf;
    }
    Float pdfPath = BalanceCancelled<2>(misPDFs, misWeights);

    // MIS of NEE
    Vector3 lightWIWorld = shadowRay.dir;
    Vector3 lightWILocal = Math::Normalize(tContext.InvApplyN(lightWIWorld));
    Ray lightWILocalRay = Ray(lightWILocal, surf.position);
    misPDFs = {mat.Pdf(lightWILocalRay, wO), mixture.Pdf(lightWIWorld)};
    Float pdfShadow = BalanceCancelled<2>(misPDFs, misWeights);
    misPDFs = {pdfShadow, lightSample.pdf};
    misWeights = {Float(1), Float(1)};
    pdfShadow = BalanceCancelled<2>(misPDFs, misWeights);


    Spectrum throughput = rayState.dThroughput[rayIndex];
    Float prevPDF       = rayState.dPDFChain[rayIndex];
    // ================== //
    //   NEE Throughput   //
    // ================== //
    Spectrum shadowRadiance = throughput;
    shadowRadiance *= mat.Evaluate(lightWILocalRay, wO).reflectance;
    shadowRadiance *= lightSample.value.emission;
    shadowRadiance = DivideByPDF(shadowRadiance, prevPDF * pdfShadow);

    // ================== //
    //   Path Throughput  //
    // ================== //
    throughput *= combinedSample.eval.reflectance;
    RayCone rayConeOut = rConeRefract.ConeAfterScatter(combinedSample.wI.dir,
                                                       surf.geoNormal);

    // ================ //
    //    Dispersion    //
    // ================ //
    if constexpr(!SpectrumConv::IsRGB)
    {
        if(combinedSample.eval.isDispersed)
        {
            spectrumConverter.DisperseWaves();
            spectrumConverter.StoreWaves();
        }
    }

    // ================ //
    // Russian Roulette //
    // ================ //
    bool isSpecular = MaterialCommon::IsSpecular(specularity);
    dataPack.depth += 1u;
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    bool isPathDead = (dataPack.depth >= rrRange[1]);
    if(!isPathDead && dataPack.depth >= rrRange[0] && !isSpecular)
    {
        Float rrXi = rng.NextFloat<Material::SampleRNList.TotalRNCount()>();
        Float rrFactor = throughput.Sum() * Float(0.33333);
        auto result = RussianRoulette(throughput, rrFactor, rrXi);
        isPathDead = !result.has_value();
        throughput = result.value_or(throughput);
    }

    // Change the ray type, if mat is highly specular
    // we do not bother casting NEE ray. So if this ray
    // hits a light somehow it should not assume MIS is enabled.
    dataPack.type = isSpecular ? RayType::SPECULAR_RAY : RayType::PATH_RAY;

    // Selectively write if path is alive
    if(!isPathDead)
    {
        // If alive update state
        rayState.dThroughput[rayIndex] = throughput;
        rayState.dPDFChain[rayIndex]   = prevPDF * pdfPath;
        rayState.dPrevMatPDF[rayIndex] = pdfPath;

        //// Store this pdf for path ray's potential light hit
        prevPDF *= pdfPath;

        // ================ //
        //  Scattered Ray   //
        // ================ //
        Vector3 nudgeNormal = surf.geoNormal;
        if(combinedSample.eval.isPassedThrough)
            nudgeNormal *= Float(-1);
        Ray rayOut = combinedSample.wI.Nudge(nudgeNormal);

        // If I remember correctly, OptiX does not like INF on rays,
        // so we put flt_max here.
        Vector2 tMMOut = Vector2(0, std::numeric_limits<Float>::max());
        RayToGMem(params.rayState.dOutRays, rayIndex, rayOut, tMMOut);
        // Continue the ray cone
        rayState.dOutRayCones[rayIndex] = rayConeOut;
    }
    else
    {
        dataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    }
    // Write the updated state back
    rayState.dPathDataPack[rayIndex] = dataPack;
}

// ================== //
//      LIGHT         //
// ================== //
template<LightGroupC L, TransformGroupC T>
MR_HF_DEF
void LightWorkFunction<L, T>::Call(const Light& l, RNGDispenser&, const SpectrumConv&,
                                   const Params& params, RayIndex rayIndex, uint32_t)
{
    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    bool switchToMISPdf = pathDataPack.type == RayType::PATH_RAY;
    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    auto [ray, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    RayCone rayCone = params.in.dRayCones[rayIndex].Advance(tMM[1]);
    if(switchToMISPdf)
    {
        using Distribution::MIS::BalanceCancelled;
        using Distribution::Common::DivideByPDF;
        //
        std::array<Float, 2> weights = {1, 1};
        std::array<Float, 2> pdfs;
        //pdfs[0] = params.rayState.dPrevMatPDF[rayIndex];
        // We need to find the index of this specific light
        // Light sampler will handle it
        HitKeyPack hitKeyPack = params.in.dKeys[rayIndex];
        MetaHit hit = params.in.dHits[rayIndex];
        pdfs[1] = params.globalState.lightSampler.PdfLight(hitKeyPack, hit, ray);
        Float misPdf = BalanceCancelled<2>(pdfs, weights);
        // We premultiply the throughput under the assumption this will not hit a light,
        // but we did. So revert the multiplication first then multiply with
        // MIS weight.
        throughput *= pdfs[0];
        throughput = DivideByPDF(throughput, misPdf);
    }

    Vector3 wO = -ray.dir;
    Spectrum emission;
    if constexpr(Light::IsPrimitiveBackedLight)
    {
        // It is more accurate to use hit if we actually hit the material
        using Hit = typename Light::Primitive::Hit;
        static constexpr uint32_t N = Hit::Dims;
        MetaHit metaHit = params.in.dHits[rayIndex];
        Hit hit = metaHit.AsVector<N>();
        emission = l.EmitViaHit(wO, hit, rayCone);
    }
    else
    {
        Vector3 position = ray.AdvancedPos(tMM[1]);
        emission = l.EmitViaSurfacePoint(wO, position, rayCone);
    }

    // Check the depth if we exceed it, do not accumulate
    // we terminate the path regardless
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    if((pathDataPack.depth + 1u) <= rrRange[1])
    {
        Spectrum radianceEstimate = emission * throughput;
        params.rayState.dPathRadiance[rayIndex] += radianceEstimate;
    }
    // Set the path as dead
    pathDataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    params.rayState.dPathDataPack[rayIndex] = pathDataPack;
}
}

//static constexpr uint64_t DATA_COUNT = 10'500'000;
//static constexpr uint64_t TABLE_SIZE = Math::NextPowerOfTwo(DATA_COUNT * 2);
//
//static constexpr auto HASH_PER_GRID = sizeof(SpatioDirCode);
//static constexpr auto DATA_PER_GRID = (sizeof(GuidedPTRDetail::MCAlber2025BaseParams) +
//                                        sizeof(GuidedPTRDetail::MarkovChainCountParam) +
//                                        sizeof(GuidedPTRDetail::IrradianceParam));
//
//static constexpr auto TOTAL_DATA_SIZE = DATA_COUNT * DATA_PER_GRID;
//static constexpr auto HASH_DATA_SIZE = DATA_COUNT * HASH_PER_GRID;
//
//static constexpr auto TOTAL_DATA_MIB = double(TOTAL_DATA_SIZE) / 1024. / 1024.;
//static constexpr auto HASH_DATA_MIB = double(HASH_DATA_SIZE) / 1024. / 1024.;
//
//static constexpr auto FULL_TOTAL_MIB = TOTAL_DATA_MIB + HASH_DATA_MIB;
//static_assert()
