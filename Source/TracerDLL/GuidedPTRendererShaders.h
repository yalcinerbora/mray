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
    struct alignas(8) MCState
    {
        Vector3  target;
        Float    cos;
        Float    weight;
    };
    using MCCount      = uint16_t;
    using MCLock       = uint32_t; // Tied to ray index

    struct alignas(8) MCIrradiance
    {
        Float irrad;
        // This is a waste, paper uses up to 1024
        // values with a 3 channel 16-bit half
        uint32_t N;

        MR_GF_DECL
        static MCIrradiance
        AtomicEMA(MCIrradiance& dLoc, Float radEst);
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
        Float       lobeProbablity = Float(0.0);
        // For debugging etc.
        DisplayMode displayMode;
    };

    using LightSampler = DirectLightSamplerUniform<MetaLightList>;

    struct GlobalState
    {
        using SpecConverterData = typename SpectrumContextJakob2019::Data;

        Vector2ui           russianRouletteRange;
        SpecConverterData   specContextData;
        LightSampler        lightSampler;
        Float               lobeProbability;
        // Hash Grid and Alber2025 data
        HashGridView        hashGrid;
        Span<MCState>       dMCStates;
        Span<MCCount>       dMCCounts;
        Span<MCLock>        dMCLocks;
        Span<MCIrradiance>  dMCIrradiances;
    };

    struct RayState
    {
        Span<Spectrum>        dPathRadiance;
        Span<ImageCoordinate> dImageCoordinates;
        Span<Float>           dFilmFilterWeights;
        Span<SpectrumWaves>   dPathWavelengths;
        // Path state
        Span<Spectrum>        dThroughputs;
        Span<PathDataPack>    dPathDataPack;
        Span<Float>           dPrevPDF;
        Span<Float>           dPrevPathReflectanceOrOutRadiance;
        Span<Float>           dScoreSums;
        // Backup RNG State
        Span<BackupRNGState>  dBackupRNGStates;
        // NEE Related
        Span<Spectrum>        dShadowRayRadiance;
        Span<RayGMem>         dShadowRays;
        Span<RayCone>         dShadowRayCones;
        Span<Float>           dShadowPrevPathReflectance;
        // Current lifted markov chain
        Span<uint32_t>        dLiftedMCIndices;
    };

    using GaussianLobe = Distribution::GaussianLobe;

    template<uint32_t N>
    struct GaussLobeMixtureT
    {
        static constexpr uint32_t LobeCount     = N;
        static constexpr auto     SampleRNList = GenRNRequestList<2>();

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
        MR_GF_DECL void             Product(const std::array<GaussianLobe, 2>& matLobes);
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
        MR_GF_DECL void             Product(const std::array<GaussianLobe, 2>& matLobes);
    };

    static constexpr uint32_t MC_LOBE_COUNT = 16;

    // TODO: Macro for CPU/GPU after profiling
    //using GaussianLobeMixture = GaussLobeMixtureSharedT<MC_LOBE_COUNT>;
    using GaussianLobeMixture = GaussLobeMixtureT<MC_LOBE_COUNT>;

    MR_HF_DECL
    GaussianLobe SufficientStatsToLobe(const MCState& state,
                                       uint32_t mcSampleCount,
                                       const Vector3& samplePos);

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType>
    struct WorkFunction
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType,
                                       SpectrumContextJakob2019, 1u);
        using Params = RenderWorkParams<GlobalState, RayState, PG, MG, TG>;

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
        using Params = RenderLightWorkParams<GlobalState, RayState, LG, TG>;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params& params, RayIndex rayIndex, uint32_t laneId);

    };

}

namespace GuidedPTRDetail
{

MR_GF_DEF
MCIrradiance
MCIrradiance::AtomicEMA(MCIrradiance& dLoc, Float radEst)
{
    // TODO: Here we do atomicCAS loop
    // this may be slow due to heavy sync (expecially on the first bounce since
    // rays are corraleted)
    return DeviceAtomic::EmulateAtomicOp(dLoc, [radEst](MCIrradiance e)
    {
        static constexpr Float MIN_EMA_RATIO_IRRAD = Float(0.01);
        static constexpr uint32_t MAX_SAMPLE = uint32_t(2048);
        //
        e.N = Math::Min(e.N + 1, MAX_SAMPLE);
        Float t = Math::Max(Float(1) / Float(e.N), MIN_EMA_RATIO_IRRAD);
        e.irrad = Math::Lerp(e.irrad, radEst, t);
        return e;
    });
}

MR_HF_DEF
GaussianLobe SufficientStatsToLobe(const MCState& state,
                                   uint32_t mcSampleCount,
                                   const Vector3& samplePos)
{
    static constexpr Float DirGuidePrior = Float(0.2);
    static constexpr Float Epsilon       = Float(0.0001);
    static constexpr Float MaxCos        = Float(1) - Epsilon;

    // Fetch direction
    Vector3 target = state.target;
    target = (state.weight > 0) ? (target / state.weight) : target;

    // Dir
    Vector3 dir = target - samplePos;
    Float distSqr = Math::LengthSqr(dir);
    Float priorState = Math::Max(Epsilon, DirGuidePrior / distSqr);
    dir /= Math::Sqrt(distSqr);

    // Kappa
    Float nSqr = Float(mcSampleCount) * Float(mcSampleCount);
    Float meanCos = nSqr * Math::Clamp(state.cos / state.weight, Float(0), MaxCos);
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
    if(sumWeight == 0) return Float(0);

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
    uint32_t liftedNodeIndex = UINT32_MAX;
    for(uint32_t i = 0; i < N; i++)
    {
        // TODO: Sample with randomness
        Vector3 posJitter = position + Vector3(rng.NextFloat());
        Vector3 nJitter   = Math::Normalize(normal + Vector3(rng.NextFloat()));
        auto [hashGridIndex, _] = gs.hashGrid.TryInsertAtomic(posJitter, nJitter);

        MCState state = gs.dMCStates[hashGridIndex];
        uint16_t mcN  = gs.dMCCounts[hashGridIndex];
        lobes[i]      = SufficientStatsToLobe(state, mcN, position);
        weights[i]    = state.weight;
        sumWeight    += state.weight;
        //
        Float sumRatio = (sumWeight != Float(0)) ? (state.weight / sumWeight) : Float(1);
        if(rng.NextFloat() < sumRatio)
        {
            sampleIndex = i;
            liftedNodeIndex = hashGridIndex;
        }
    }
    return liftedNodeIndex;
}

template<uint32_t N>
MR_GF_DEF void GaussLobeMixtureT<N>::Product(const std::array<GaussianLobe, 2>& matLobes)
{
    for(uint32_t i = 0; i < N; i++)
        lobes[i] = lobes[i].Product(matLobes[0]).Product(matLobes[1]);
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

        MCState state = gs.dMCStates[hashGridIndex];
        uint16_t mcN  = gs.dMCCounts[hashGridIndex];
        sumWeight    += state.weight;

        auto lobe = SufficientStatsToLobe(state, mcN, position);
        SMem().sLobeX[Index(i)] = lobe.dir[0];
        SMem().sLobeY[Index(i)] = lobe.dir[1];
        SMem().sLobeZ[Index(i)] = lobe.dir[2];
        SMem().sLobeKappa[Index(i)] = lobe.kappa;
        SMem().sLobeAlpha[Index(i)] = lobe.alpha;
        SMem().sLobeWeights[Index(i)] = state.weight;

        Float sumRatio = (sumWeight != Float(0)) ? (state.weight / sumWeight) : Float(1);
        if(rng.NextFloat() < sumRatio)
        {
            sampleIndex = i;
            liftedNodeIndex = hashGridIndex;
        }
    }
    return liftedNodeIndex;
}

template<uint32_t N>
MR_GF_DEF void GaussLobeMixtureSharedT<N>::Product(const std::array<GaussianLobe, 2>& matLobes)
{
    for(uint32_t i = 0; i < N; i++)
    {
        GaussianLobe l(Vector3(SMem().sLobeX[Index(i)],
                               SMem().sLobeY[Index(i)],
                               SMem().sLobeZ[Index(i)]),
                       SMem().sLobeKappa[Index(i)],
                       SMem().sLobeAlpha[Index(i)]);
        l = l.Product(matLobes[0]).Product(matLobes[1]);

        SMem().sLobeX[Index(i)] = l.dir[0];
        SMem().sLobeY[Index(i)] = l.dir[1];
        SMem().sLobeZ[Index(i)] = l.dir[2];
        SMem().sLobeKappa[Index(i)] = l.kappa;
        SMem().sLobeAlpha[Index(i)] = l.alpha;
    }

}

// ================== //
//        PATH        //
// ================== //
template<PrimitiveGroupC P, MaterialGroupC M, TransformGroupC T>
MR_GF_DEF
void WorkFunction<P, M, T>::Call(const Primitive&, const Material& mat, const Surface& surf,
                                 const RayConeSurface& surfRayCone, const TContext& tContext,
                                 SpectrumConv& spectrumConverter, RNGDispenser& rng, const Params& params,
                                 RayIndex rayIndex, uint32_t)
{
    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    using Distribution::MIS::BalanceCancelled;
    static constexpr auto MISSampleStart = Math::Max(GaussianLobeMixture::SampleRNList.TotalRNCount(),
                                                     Material::SampleRNList.TotalRNCount());
    static constexpr auto RRSampleStart = MISSampleStart + 1;
    static constexpr auto LightSampleStart = RRSampleStart + 1;


    const GlobalState& gs = params.globalState;
    const RayState& rs    = params.rayState;
    BackupRNG backupRNG   = BackupRNG(rs.dBackupRNGStates[rayIndex]);
    PathDataPack dataPack = rs.dPathDataPack[rayIndex];
    if(dataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    auto [rayIn, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    Vector3 wOWorld = Math::Normalize(-rayIn.dir);
    Vector3 wO = Math::Normalize(tContext.InvApplyV(wOWorld));
    RayConeSurface rConeRefract = mat.RefractRayCone(surfRayCone, wO);
    Spectrum throughput = rs.dThroughputs[rayIndex];
    Float specularity = mat.Specularity();

    // ====================== //
    //     Load Mixture       //
    // ====================== //
    GaussianLobeMixture mixture;
    uint32_t liftedMCIndex = mixture.LoadStochastic(surf.position,
                                                    surf.geoNormal,
                                                    backupRNG,
                                                    gs);
    // TODO: Product sampling
    //std::array<GaussianLobe, 2> materialLobes =
    //{
    //    GaussianLobe(surf.geoNormal, Float(0.1)),
    //    GaussianLobe(Graphics::Reflect(surf.geoNormal, wOWorld), Float(specularity) * Float(100))
    //};
    //mixture.Product(materialLobes);

    // ====================== //
    // Sample Material or vMF //
    // ====================== //
    std::array<Float, 2> misPDFs = {};
    std::array<Float, 2> misWeights = {};
    if(!MaterialCommon::IsSpecular(specularity))
        misWeights[0] = Math::Lerp(Float(0), gs.lobeProbability, specularity);
    misWeights[1] = Float(1) - misWeights[0];

    BxDFSample pathSample;
    if(rng.NextFloat<MISSampleStart>() < misWeights[0])
    {
        assert(false);
        // Sample Mixture
        auto mixtureSample = mixture.Sample(rng);
        Vector3 wILocal = Math::Normalize(tContext.InvApplyV(mixtureSample.value));
        pathSample.wI = Ray(mixtureSample.value, surf.position);
        pathSample.eval = mat.Evaluate(Ray(wILocal, surf.position), wO);
        misPDFs[0] = mixtureSample.pdf;
        misPDFs[1] = mat.Pdf(Ray(wILocal, surf.position), wO);
    }
    else
    {
        // Sample Material
        pathSample = mat.SampleBxDF(wO, rng);
        pathSample.wI.dir = Math::Normalize(tContext.ApplyV(pathSample.wI.dir));
        misPDFs[0] = mixture.Pdf(pathSample.wI.dir);
        misPDFs[1] = pathSample.pdf;
    }
    Float pdfPath = BalanceCancelled<2>(misPDFs, misWeights);

    // ================== //
    //   Path Throughput  //
    // ================== //
    Spectrum pathThroughput = throughput * pathSample.eval.reflectance;
    RayCone pathRayConeOut = rConeRefract.ConeAfterScatter(pathSample.wI.dir,
                                                           surf.geoNormal);
    // ================ //
    // Russian Roulette //
    // ================ //
    bool isSpecular = MaterialCommon::IsSpecular(specularity);
    dataPack.depth += 1u;
    Vector2ui rrRange = gs.russianRouletteRange;
    bool isPathDead = (dataPack.depth >= rrRange[1]);
    if(!isPathDead && dataPack.depth >= rrRange[0] && !isSpecular)
    {
        Float rrXi = rng.NextFloat<RRSampleStart>();
        Float rrFactor = pathThroughput.Sum() * Float(0.33333);
        auto result = RussianRoulette(pathThroughput, rrFactor, rrXi);
        isPathDead = !result.has_value();
        pathThroughput = result.value_or(pathThroughput);
    }

    // Change the ray type, if mat is highly specular
    // we do not bother casting NEE ray. So if this ray
    // hits a light somehow it should not assume MIS is enabled.
    dataPack.type = isSpecular ? RayType::SPECULAR_RAY : RayType::PATH_RAY;

    // ================ //
    //    Dispersion    //
    // ================ //
    static constexpr Float SpectrumCountInv = Float(1) / Float(SpectraPerSpectrum);
    Float avgPathReflectance;
    if constexpr(!SpectrumConv::IsRGB)
    {
        // Path
        if(pathSample.eval.isDispersed)
        {
            spectrumConverter.DisperseWaves();
            spectrumConverter.StoreWaves();
            avgPathReflectance = pathSample.eval.reflectance[0];
        }
        else avgPathReflectance = pathSample.eval.reflectance.Sum() * SpectrumCountInv;
    }

    // Selectively write if path is alive
    if(!isPathDead)
    {
        // ================ //
        //  Scattered Ray   //
        // ================ //
        Vector3 nudgeNormal = surf.geoNormal;
        if(pathSample.eval.isPassedThrough)
            nudgeNormal *= Float(-1);
        Ray pathRayOut = pathSample.wI.Nudge(nudgeNormal);
        // If I remember correctly, OptiX does not like INF on rays,
        // so we put flt_max here.
        Vector2 pathTMMOut = Vector2(MathConstants::LargeEpsilon<Float>(),
                                     std::numeric_limits<Float>::max());
        RayToGMem(params.common.dRays, rayIndex, pathRayOut, pathTMMOut);
        params.common.dRayCones[rayIndex] = pathRayConeOut;
        // We prematurely divide by the PDF as if the path will not hit a light
        // but we store current PDF as well, so that if path hits a light,
        // it can readjust its weight via MIS.
        rs.dThroughputs[rayIndex] = DivideByPDF(pathThroughput, pdfPath);
        rs.dPrevPDF[rayIndex] = pdfPath;
        // When we update the MC, this will be used to estimate radiant exitance
        // and MC state termination update etc.
        rs.dPrevPathReflectanceOrOutRadiance[rayIndex] = DivideByPDF(avgPathReflectance,
                                                                     pdfPath);
        // Save the score sum for MC selection as well
        rs.dScoreSums[rayIndex] = mixture.sumWeight;
        rs.dLiftedMCIndices[rayIndex] = liftedMCIndex;
    }

    // Do not bother with NEE if specular or path is dead
    if(isPathDead || isSpecular)
    {
        dataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
        // Write the updated state back
        rs.dPathDataPack[rayIndex] = dataPack;
        return;
    }

    // We do two sample mis
    // NEE / MIS of Lobes and BxDF
    // ====================== //
    //          NEE           //
    // ====================== //
    rng.Advance(LightSampleStart);
    LightSample lightSample = gs.lightSampler.SampleLight(rng, spectrumConverter,
                                                          surf.position,
                                                          surf.geoNormal,
                                                          rConeRefract);
    auto [shadowRay, shadowTMM] = lightSample.value.SampledRay(surf.position);

    // MIS of NEE
    Ray wIShadow = tContext.InvApply(shadowRay);
    misPDFs = {mixture.Pdf(shadowRay.dir), mat.Pdf(wIShadow, wO)};
    Float pdfShadow = BalanceCancelled<2>(misPDFs, misWeights);
    misPDFs = {pdfShadow, lightSample.pdf};
    misWeights = {Float(1), Float(1)};
    pdfShadow = BalanceCancelled<2>(misPDFs, misWeights);

    // ================== //
    //   NEE Throughput   //
    // ================== //
    Spectrum shadowRadiance = throughput;
    BxDFEval shadowRayEval = mat.Evaluate(wIShadow, wO);
    shadowRadiance *= shadowRayEval.reflectance;
    shadowRadiance *= lightSample.value.emission;
    shadowRadiance = DivideByPDF(shadowRadiance, pdfShadow);
    RayCone shadowRayConeOut = rConeRefract.ConeAfterScatter(shadowRay.dir,
                                                             surf.geoNormal);

    // ================ //
    //    Dispersion    //
    // ================ //
    Float avgShadowReflectance;
    if constexpr(!SpectrumConv::IsRGB)
    {
        // NEE
        if(shadowRayEval.isDispersed)
        {
            Float first = shadowRadiance[0];
            shadowRadiance = Spectrum::Zero();
            shadowRadiance[0] = first;
            avgShadowReflectance = shadowRayEval.reflectance[0];
        }
        else avgShadowReflectance = shadowRayEval.reflectance.Sum() * SpectrumCountInv;
    }

    // ==================== //
    //  Shadow(NEE) Ray Out //
    // ==================== //
    Vector3 nudgeNormal = surf.geoNormal;
    if(shadowRayEval.isPassedThrough)
        nudgeNormal *= Float(-1);
    shadowRay = shadowRay.Nudge(nudgeNormal);
    RayToGMem(rs.dShadowRays, rayIndex, shadowRay, shadowTMM);
    rs.dShadowRayCones[rayIndex] = shadowRayConeOut;
    rs.dShadowRayRadiance[rayIndex] = shadowRadiance;
    rs.dShadowPrevPathReflectance[rayIndex] = DivideByPDF(avgShadowReflectance,
                                                          pdfShadow);
    // Write the updated state back
    rs.dPathDataPack[rayIndex] = dataPack;
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
    auto [ray, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    RayCone rayCone = params.common.dRayCones[rayIndex].Advance(tMM[1]);
    Spectrum throughput = params.rayState.dThroughputs[rayIndex];
    if(switchToMISPdf)
    {
        using Distribution::MIS::BalanceCancelled;
        using Distribution::Common::DivideByPDF;
        //
        std::array<Float, 2> weights = {1, 1};
        std::array<Float, 2> pdfs;
        pdfs[0] = params.rayState.dPrevPDF[rayIndex];
        // We need to find the index of this specific light
        // Light sampler will handle it
        HitKeyPack hitKeyPack = params.common.dKeys[rayIndex];
        MetaHit hit = params.common.dHits[rayIndex];
        pdfs[1] = params.globalState.lightSampler.PdfLight(hitKeyPack, hit, ray);
        Float misPdf = BalanceCancelled<2>(pdfs, weights);

        // We pre-divided the throughput under the assumption this
        // will not hit a light, but we did. So revert the division
        // first then multiply with MIS weight.
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
        MetaHit metaHit = params.common.dHits[rayIndex];
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
