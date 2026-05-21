#pragma once

#include <array>

#include "Core/NamedEnum.h"

#include "Tracer/PrimitiveC.h"
#include "Tracer/MaterialC.h"
#include "Tracer/RenderWork.h"
#include "Tracer/DistributionFunctions.h"
#include "Tracer/LightSampler.h"
#include "Tracer/SpectrumContext.h"
#include "Tracer/PathTracerRendererBase.h"

template<SpectrumContextC SC>
class PathTracerRendererT;

namespace PathTraceRDetail
{
    enum class SampleModeEnum
    {
        PURE,
        NEE,
        NEE_WITH_MIS,
        //
        END
    };
    inline constexpr std::array SampleModeNames =
    {
        "Pure",
        "WithNextEventEstimation",
        "WithNEEAndMIS"
    };
    using SampleMode = NamedEnum<SampleModeEnum, SampleModeNames>;

    struct Options
    {
        uint32_t            totalSPP = 16'384;
        uint32_t            burstSize = 1;
        Vector2ui           russianRouletteRange = Vector2ui(4, 20);
        LightSamplerType    lightSampler = LightSamplerType::E::UNIFORM;
        SampleMode          sampleMode = SampleMode::E::PURE;
        RenderMode          renderMode = RenderMode::E::THROUGHPUT;
        bool                sampleMedia = false;
    };

    template<class LightSampler, class SpectrumConverter>
    struct GlobalState
    {
        using SpecConverterData = typename SpectrumConverter::Data;

        Vector2ui           russianRouletteRange;
        SampleMode          sampleMode;
        LightSampler        lightSampler;
        SpecConverterData   specContextData;
        bool                sampleMedia;
    };

    struct RayState
    {
        // Output related
        Span<Spectrum>          dPathRadiance;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
        // Path state
        Span<Spectrum>          dThroughput;
        Span<PathDataPack>      dPathDataPack;
        // May be empty if not spectral
        Span<SpectrumWaves>     dPathWavelengths;

        // Only used when NEE/MIS is active
        Span<RayGMem>           dShadowRays;
        Span<RayCone>           dShadowRayCones;
        Span<Spectrum>          dShadowRayRadiance;

        // ========================================== //
        // Only used when media sampling is DISABLED  //
        // and NEE/MIS is active                      //
        // ========================================== //
        // We can get away with a single float here
        // since we do not do per-channel MIS
        // and we can embed/recalculate the other PDF
        // when we collapse the ray (when it hits the light)
        Span<Float>             dPrevMatPDF;

        // ========================================== //
        // Only used when media sampling is ENABLED   //
        // and NEE & MIS is active.                   //
        // ========================================== //
        // This is kind of costly since we can't collapse
        // pdfs.
        Span<Spectrum>          dRPathPDF;
        Span<Spectrum>          dRLightPDF;
        Span<Spectrum>          dRPathPDFShadow;
        Span<Spectrum>          dRLightPDFShadow;

        // Volume rendering related (Common)
        Span<BackupRNGState>    dBackupRNGStates;
        Span<RayMediaListPack>  dMediaListPack;
        Span<RayMediaListPack>  dShadowMediaListPack;
    };

    template<class LightSampler, class SpectrumConverter, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkParams = RenderWorkParams
    <
        GlobalState<LightSampler, SpectrumConverter>,
        RayState,
        PG, MG, TG
    >;
    template<class LightSampler, class SpectrumConverter, LightGroupC LG, TransformGroupC TG>
    using LightWorkParams = RenderLightWorkParams
    <
        GlobalState<LightSampler, SpectrumConverter>,
        RayState,
        LG, TG
    >;
    template<class LightSampler, class SpectrumConverter, MediumGroupC MG, TransformGroupC TG>
    using MediumWorkParams = RenderMediumWorkParams
    <
        GlobalState<LightSampler, SpectrumConverter>,
        RayState,
        MG, TG
    >;

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType>
    struct WorkFunction
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType, SpectrumCtxType, 1u);
        using Params = WorkParams<EmptyType, SpectrumConv, PG, MG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<EmptyType, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Primitive&, const Material&, const Surface&,
                         const RayConeSurface&, const TContext&,
                         SpectrumConv&, RNGDispenser&,
                         const Params& params,
                         RayIndex rayIndex, uint32_t laneId);
    };

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType, class LightSampler>
    struct WorkFunctionNEE
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType, SpectrumCtxType, 1u);
        using Params      = WorkParams<LightSampler, SpectrumConv, PG, MG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<LightSampler, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Primitive&, const Material&, const Surface&,
                         const RayConeSurface&, const TContext&,
                         const SpectrumConv&, RNGDispenser&,
                         const Params& params,
                         RayIndex rayIndex, uint32_t laneId);
    };

    template<LightGroupC LGType, TransformGroupC TGType, class SpectrumCtxType>
    struct LightWorkFunction
    {
        MRAY_LIGHT_WORK_FUNCTOR_DEFINE_TYPES(LGType, TGType, SpectrumCtxType, 1u);
        using Params      = LightWorkParams<EmptyType, SpectrumConv, LG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<EmptyType, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params&, RayIndex rayIndex, uint32_t laneId);

    };

    template<LightGroupC LGType, TransformGroupC TGType,
             class SpectrumCtxType, class LightSampler>
    struct LightWorkFunctionWithNEE
    {
        MRAY_LIGHT_WORK_FUNCTOR_DEFINE_TYPES(LGType, TGType, SpectrumCtxType, 1u);
        using Params      = LightWorkParams<LightSampler, SpectrumConv, LG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<LightSampler, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params& params, RayIndex rayIndex, uint32_t laneId);
    };

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType, class LightSampler>
    struct WorkFunctionMedia
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType, SpectrumCtxType, 1u);
        using Params      = WorkParams<LightSampler, SpectrumConv, PG, MG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<LightSampler, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Primitive&, const Material&, const Surface&,
                         const RayConeSurface&, const TContext&,
                         SpectrumConv&, RNGDispenser&,
                         const Params& params,
                         RayIndex rayIndex, uint32_t laneId);
    };

    template<LightGroupC LGType, TransformGroupC TGType,
             class SpectrumCtxType, class LightSampler>
    struct LightWorkFunctionMedia
    {
        MRAY_LIGHT_WORK_FUNCTOR_DEFINE_TYPES(LGType, TGType, SpectrumCtxType, 1u);
        using Params      = LightWorkParams<LightSampler, SpectrumConv, LG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<LightSampler, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params& params, RayIndex rayIndex, uint32_t laneId);
    };

    template<MediumGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType>
    struct MediumWorkFunction
    {
        MRAY_MEDIUM_WORK_FUNCTOR_DEFINE_TYPES(MGType, TGType, SpectrumCtxType, 1u);
        using Params      = MediumWorkParams<EmptyType, SpectrumConv, MG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<EmptyType, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Medium&, const TContext&, SpectrumConv&,
                         RNGDispenser&, const Params&, RayIndex, uint32_t);
    };

    template<MediumGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType, class LightSampler>
    struct MediumWorkFunctionWithNEE
    {
        MRAY_MEDIUM_WORK_FUNCTOR_DEFINE_TYPES(MGType, TGType, SpectrumCtxType, 1u);
        using Params      = MediumWorkParams<LightSampler, SpectrumConv, MG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<LightSampler, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Medium&, const TContext&, SpectrumConv&,
                         RNGDispenser&, const Params&, RayIndex, uint32_t);
    };

    template<MediumGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType>
    struct MediumWorkFunctionTransmittance
    {
        MRAY_MEDIUM_WORK_FUNCTOR_DEFINE_TYPES(MGType, TGType, SpectrumCtxType, 1u);
        using Params      = MediumWorkParams<EmptyType, SpectrumConv, MG, TG>;
        using GlobalState = PathTraceRDetail::GlobalState<EmptyType, SpectrumConv>;
        using RayState    = PathTraceRDetail::RayState;

        MR_HF_DECL
        static void Call(const Medium&, const TContext&, SpectrumConv&,
                         RNGDispenser&, const Params&, RayIndex, uint32_t);
    };
}

namespace PathTraceRDetail
{

// ======================== //
//     PURE PATH TRACE      //
// ======================== //
template<PrimitiveGroupC P, MaterialGroupC M, TransformGroupC T, class SC>
MR_HF_DEF
void WorkFunction<P, M, T, SC>::Call(const Primitive&, const Material& mat, const Surface& surf,
                                     const RayConeSurface& surfRayCone, const TContext& tContext,
                                     SpectrumConv& spectrumConverter, RNGDispenser& rng, const Params& params,
                                     RayIndex rayIndex, uint32_t)
{
    PathDataPack dataPack = params.rayState.dPathDataPack[rayIndex];
    if(dataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    // ================ //
    // Sample Material  //
    // ================ //
    auto [rayIn, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    Vector3 wO = Math::Normalize(tContext.InvApplyV(-rayIn.dir));
    RayConeSurface rConeRefract = mat.RefractRayCone(surfRayCone, wO);
    BxDFSample raySample = mat.SampleBxDF(wO, rng);
    raySample.wI.dir = Math::Normalize(tContext.ApplyV(raySample.wI.dir));

    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    throughput *= raySample.eval.reflectance;

    RayCone rayConeOut = rConeRefract.ConeAfterScatter(raySample.wI.dir,
                                                       surf.geoNormal);

    // ================ //
    //    Dispersion    //
    // ================ //
    if constexpr(!SpectrumConv::IsRGB)
    {
        if(raySample.eval.isDispersed)
        {
            spectrumConverter.DisperseWaves();
            spectrumConverter.StoreWaves();
        }
    }

    // ================ //
    // Russian Roulette //
    // ================ //
    Float specularity = mat.Specularity();
    dataPack.depth += 1u;
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    bool isPathDead = (dataPack.depth >= rrRange[1]);
    if(!isPathDead && dataPack.depth >= rrRange[0] &&
       !MaterialCommon::IsSpecular(specularity))
    {
        // TODO: This is still wrong maybe? What about dispersion?
        static constexpr Float ChannelCountInv = (SpectrumConv::IsRGB)
            ? Float(0.33333333)
            : Float(1) / Float(SpectraPerSpectrum);

        Float rrXi = rng.NextFloat<Material::SampleRNList.TotalRNCount()>();
        Float rrFactor = throughput.Sum() * ChannelCountInv;
        auto result = RussianRoulette(throughput, rrFactor, rrXi);
        isPathDead = !result.HasValue();
        throughput = result.ValueOr(throughput);
    }

    // Change the ray type, if mat is highly specular
    // we do not bother casting NEE ray. So if this ray
    // hits a light somehow it should not assume MIS is enabled.
    bool isSpecular = MaterialCommon::IsSpecular(specularity);
    dataPack.type = isSpecular ? RayType::SPECULAR_RAY : RayType::PATH_RAY;

    // Selectively write if path is alive
    if(!isPathDead)
    {
        if(params.globalState.sampleMedia)
            params.rayState.dMediaListPack[rayIndex].SetPassthrough(raySample.eval.isPassedThrough);

        // If alive update throughput
        params.rayState.dThroughput[rayIndex] = DivideByPDF(throughput, raySample.pdf);

        // Save the previous pdf (aka. current pdf, naming is for the user)
        // To correctly do MIS we will need it
        if(params.globalState.sampleMode == SampleMode::E::NEE_WITH_MIS)
            params.rayState.dPrevMatPDF[rayIndex] = raySample.pdf;

        // ================ //
        //  Scattered Ray   //
        // ================ //
        Vector3 nudgeNormal = surf.geoNormal;
        if(raySample.eval.isPassedThrough)
            nudgeNormal *= Float(-1);
        Ray rayOut = raySample.wI.Nudge(nudgeNormal);

        // If I remember correctly, OptiX does not like INF on rays,
        // so we put flt_max here.
        Vector2 tMMOut = Vector2(MathConstants::LargeEpsilon<Float>(),
                                 std::numeric_limits<Float>::max());
        RayToGMem(params.common.dRays, rayIndex, rayOut, tMMOut);
        // Continue the ray cone
        params.common.dRayCones[rayIndex] = rayConeOut;
    }
    else
    {
        dataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    }
    // Write the updated state back
    params.rayState.dPathDataPack[rayIndex] = dataPack;
}

// ======================== //
//   PURE PATH TRACE LIGHT  //
// ======================== //
template<LightGroupC L, TransformGroupC T, class SC>
MR_HF_DEF
void LightWorkFunction<L, T, SC>::Call(const Light& l, RNGDispenser&, const SpectrumConv&,
                                       const Params& params, RayIndex rayIndex, uint32_t)
{
    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    auto [ray, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    Vector3 wO = -ray.dir;
    RayCone rayCone = params.common.dRayCones[rayIndex].Advance(tMM[1]);

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

    // Check the depth if we exceed it, do not accumulate.
    // We terminate the path regardless
    pathDataPack.depth += 1u;
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    if(pathDataPack.depth <= rrRange[1])
    {
        Spectrum throughput = params.rayState.dThroughput[rayIndex];
        Spectrum radianceEstimate = emission * throughput;
        params.rayState.dPathRadiance[rayIndex] = radianceEstimate;
    }
    // Set the path as dead
    pathDataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    params.rayState.dPathDataPack[rayIndex] = pathDataPack;
}

// =============================== //
//    NEE AND/OR MIS EXTENSIONS    //
// =============================== //
template<PrimitiveGroupC P, MaterialGroupC M, TransformGroupC T, class SC, class LS>
MR_HF_DEF
void WorkFunctionNEE<P, M, T, SC, LS>::Call(const Primitive&, const Material& mat, const Surface& surf,
                                            const RayConeSurface& surfRayCone, const TContext& tContext,
                                            const SpectrumConv& specConverter, RNGDispenser& rng,
                                            const Params& params, RayIndex rayIndex, uint32_t)
{
    using LightSampler = LS;

    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    using Distribution::Common::DivideByPDF;
    // ================ //
    //       NEE        //
    // ================ //
    auto [rayIn, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    Vector3 wO = Math::Normalize(tContext.InvApplyV(-rayIn.dir));
    const LightSampler& lightSampler = params.globalState.lightSampler;
    Vector3 worldPos = surf.position;
    RayConeSurface rConeRefract = mat.RefractRayCone(surfRayCone, wO);
    LightSample lightSample = lightSampler.SampleLight(rng, specConverter,
                                                       worldPos, surf.geoNormal,
                                                       rConeRefract);
    auto [shadowRay, shadowTMM] = lightSample.value.SampledRay(worldPos);
    Ray wI = tContext.InvApply(shadowRay);

    BxDFEval matEval = mat.Evaluate(wI, wO);
    Spectrum reflectance = matEval.reflectance;
    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    throughput *= reflectance;

    // Either do MIS or normal sampling
    Float pdf;
    if(params.globalState.sampleMode == SampleMode::E::NEE_WITH_MIS)
    {
        using Distribution::MIS::BalanceCancelled;
        Float bxdfPdf = mat.Pdf(wI, wO);
        Array<Float, 2> pdfs = {bxdfPdf, lightSample.pdf};
        Array<Float, 2> weights = {Float(1), Float(1)};
        pdf = BalanceCancelled<2>(pdfs, weights);
    }
    else pdf = lightSample.pdf;
    // Pre-calculate the result, we will only do a visibility
    // check and if it succeeds we accumulate later
    Spectrum shadowRadiance = throughput * lightSample.value.emission;
    shadowRadiance = DivideByPDF(shadowRadiance, pdf);

    // ================ //
    //    Dispersion    //
    // ================ //
    if constexpr(!SpectrumConv::IsRGB)
    {
        if(matEval.isDispersed)
        {
            Float first = shadowRadiance[0];
            shadowRadiance = Spectrum::Zero();
            shadowRadiance[0] = first;
        }
    }

    // Writing
    if(MaterialCommon::IsSpecular(mat.Specularity()))
    {
        // Set the shadow ray as specular ray, we overwrite the ray state
        // but ray state is important for rays that hit light.
        // And next call (material-related one) will overwrite it
        // anyway.
        pathDataPack.type = RayType::SPECULAR_RAY;
    }
    else
    {
        // TODO: We need to check if material is transmissive
        // If transmissive we can set the geoNormal towards the
        // shadow ray and nudge
        Vector3 nudgeNormal = surf.geoNormal;
        if(matEval.isPassedThrough)
            nudgeNormal *= Float(-1);
        shadowRay = shadowRay.Nudge(nudgeNormal);
        RayToGMem(params.rayState.dShadowRays, rayIndex,
                  shadowRay, shadowTMM);
        RayCone rayConeOut = rConeRefract.ConeAfterScatter(shadowRay.dir,
                                                           surf.geoNormal);
        params.rayState.dShadowRayCones[rayIndex] = rayConeOut;
        // We can't overwrite the path throughput,
        // we will need it on next iteration
        params.rayState.dShadowRayRadiance[rayIndex] = shadowRadiance;
        pathDataPack.type = RayType::SHADOW_RAY;
    }
    params.rayState.dPathDataPack[rayIndex] = pathDataPack;
}

// ======================== //
// NEE/MIS PATH TRACE LIGHT //
// ======================== //
template<LightGroupC L, TransformGroupC T, class SC, class LS>
MR_HF_DEF
void LightWorkFunctionWithNEE<L, T, SC, LS>::Call(const Light& l, RNGDispenser&, const SpectrumConv&,
                                                  const Params& params, RayIndex rayIndex, uint32_t)
{
    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    // If mode is NEE, only camera rays are allowed
    if(params.globalState.sampleMode == SampleMode::E::NEE &&
       (pathDataPack.type != RayType::CAMERA_RAY) &&
       (pathDataPack.type != RayType::SPECULAR_RAY))
    {
        pathDataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
        params.rayState.dPathDataPack[rayIndex] = pathDataPack;
        return;
    }

    bool switchToMISPdf = (params.globalState.sampleMode == SampleMode::E::NEE_WITH_MIS &&
                           pathDataPack.type == RayType::PATH_RAY);
    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    auto [ray, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    RayCone rayCone = params.common.dRayCones[rayIndex].Advance(tMM[1]);
    if(switchToMISPdf)
    {
        using Distribution::MIS::BalanceCancelled;
        using Distribution::Common::DivideByPDF;
        //
        Array<Float, 2> weights = {Float(1), Float(1)};
        Array<Float, 2> pdfs;
        pdfs[0] = params.rayState.dPrevMatPDF[rayIndex];
        // We need to find the index of this specific light
        // Light sampler will handle it
        HitKeyPack hitKeyPack = params.common.dKeys[rayIndex];
        MetaHit hit = params.common.dHits[rayIndex];
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

// =============================== //
//                                 //
//  In media tracking mode, we     //
//  disregard NEE/MIS flags, and   //
//  these are always on.           //
//                                 //
// =============================== //
//        MEDIA SURFACE            //
// =============================== //
template<PrimitiveGroupC P, MaterialGroupC M, TransformGroupC T, class SC, class LS>
MR_HF_DEF
void WorkFunctionMedia<P, M, T, SC, LS>::Call(const Primitive&, const Material& mat, const Surface& surf,
                                              const RayConeSurface& surfRayCone, const TContext& tContext,
                                              SpectrumConv& spectrumConverter, RNGDispenser& rng,
                                              const Params& params, RayIndex rayIndex, uint32_t)
{
    using LightSampler = LS;
    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    using Distribution::MIS::BalanceCancelled;

    const auto& rS = params.rayState;
    const auto& gS = params.globalState;
    const auto& cS = params.common;
    //
    const Spectrum throughput        = rS.dThroughput[rayIndex];
    const Vector3 worldPos           = surf.position;
    auto [rayIn, tMM]                = RayFromGMem(cS.dRays, rayIndex);
    const Vector3 wO                 = Math::Normalize(tContext.InvApplyV(-rayIn.dir));
    const Spectrum rPath             = rS.dRPathPDF[rayIndex];
    const Spectrum rLight            = rS.dRLightPDF[rayIndex];
    const LightSampler& lightSampler = gS.lightSampler;
    RayConeSurface rConeRefract      = mat.RefractRayCone(surfRayCone, wO);
    PathDataPack dataPack            = rS.dPathDataPack[rayIndex];
    RayMediaListPack mediaPack       = rS.dMediaListPack[rayIndex];
    RayMediaListPack shadowMediaPack = mediaPack;
    if(dataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;
    // ================ //
    // Sample Material  //
    // ================ //
    BxDFSample pathRaySample = mat.SampleBxDF(wO, rng);
    pathRaySample.wI.dir = Math::Normalize(tContext.ApplyV(pathRaySample.wI.dir));
    Spectrum pathThroughput = throughput * pathRaySample.eval.reflectance;
    pathThroughput = DivideByPDF(pathThroughput, pathRaySample.pdf);
    // Do nothing here we scaled already (the throughput)
    // We divide the the throughput with pdf to keep the number
    // numerically calm. (Assume we multiplied and divided rPath with
    // "pathRaySample.pdf")
    Spectrum rPathOut = rPath;
    // Here we divide the "pathRaySample.pdf" so that these properly cancel
    // out with Balanced MIS (i.e, x/a / ((rl * a / a) + (ru / a) =
    // x / rl + ru))
    Spectrum rLightOut = DivideByPDF(rPath, pathRaySample.pdf);
    RayCone pathRayConeOut = rConeRefract.ConeAfterScatter(pathRaySample.wI.dir,
                                                           surf.geoNormal);
    // ================ //
    //    Dispersion    //
    // ================ //
    if constexpr(!SpectrumConv::IsRGB)
    {
        if(pathRaySample.eval.isDispersed)
        {
            spectrumConverter.DisperseWaves();
            spectrumConverter.StoreWaves();
        }
    }

    // ================ //
    // Russian Roulette //
    // ================ //
    Float specularity = mat.Specularity();
    dataPack.depth += 1u;
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    bool isPathDead = (dataPack.depth >= rrRange[1]);
    if(!isPathDead && dataPack.depth >= rrRange[0] &&
       !MaterialCommon::IsSpecular(specularity))
    {
        // TODO: This is still wrong maybe? What about dispersion?
        static constexpr Float ChannelCountInv = (SpectrumConv::IsRGB)
            ? Float(0.33333333)
            : Float(1) / Float(SpectraPerSpectrum);

        Float rrXi = rng.NextFloat<Material::SampleRNList.TotalRNCount()>();
        Float rrFactor = pathThroughput.Sum() * ChannelCountInv;
        auto result = RussianRoulette(pathThroughput, rrFactor, rrXi);
        isPathDead = !result.HasValue();
        pathThroughput = result.ValueOr(pathThroughput);
    }

    // Change the ray type, if mat is highly specular
    // we do not bother casting NEE ray. So if this ray
    // hits a light somehow it should not assume MIS is enabled.
    bool isSpecular = MaterialCommon::IsSpecular(specularity);
    dataPack.type = isSpecular ? RayType::SPECULAR_RAY : RayType::PATH_RAY;

    // Path Ray Write
    if(!isPathDead)
    {
        mediaPack.SetPassthrough(pathRaySample.eval.isPassedThrough);
        rS.dMediaListPack[rayIndex] = mediaPack;

        Vector3 nudgeNormal = surf.geoNormal;
        //
        if(pathRaySample.eval.isPassedThrough) nudgeNormal *= Float(-1);
        //
        pathRaySample.wI = pathRaySample.wI.Nudge(nudgeNormal);
        RayToGMem(cS.dRays, rayIndex, pathRaySample.wI, Vector2(0, FLT_MAX));
        cS.dRayCones[rayIndex]   = pathRayConeOut;
        rS.dRPathPDF[rayIndex]   = rPathOut;
        rS.dRLightPDF[rayIndex]  = rLightOut;
        rS.dThroughput[rayIndex] = pathThroughput;
    }

    // ================ //
    //       NEE        //
    // ================ //
    LightSample lightSample = lightSampler.SampleLight(rng, spectrumConverter,
                                                       worldPos, surf.geoNormal,
                                                       rConeRefract);
    auto [shadowRay, shadowTMM] = lightSample.value.SampledRay(worldPos);
    Ray shadowWI = tContext.InvApply(shadowRay);
    BxDFEval shadowMatEval = mat.Evaluate(shadowWI, wO);
    Spectrum shadowReflectance = shadowMatEval.reflectance;
    Spectrum shadowThroughput = throughput * shadowReflectance;
    Float shadowBXDFPdf = mat.Pdf(shadowWI, wO);
    shadowThroughput = DivideByPDF(shadowThroughput, shadowBXDFPdf);
    // Re-calculate path/light pdfs and store
    Spectrum rPathShadow = DivideByPDF(rPath * shadowBXDFPdf, shadowBXDFPdf);
    Spectrum rLightShadow = DivideByPDF(rPath * lightSample.pdf, shadowBXDFPdf);
    // Pre-calculate throughput
    Spectrum shadowRadiance = shadowThroughput * lightSample.value.emission;

    // ================ //
    //  NEE Dispersion  //
    // ================ //
    if constexpr(!SpectrumConv::IsRGB)
    {
        if(shadowMatEval.isDispersed)
        {
            Float first = shadowRadiance[0];
            shadowRadiance = Spectrum::Zero();
            shadowRadiance[0] = first;
        }
    }

    // Shadow Ray Write
    if(MaterialCommon::IsSpecular(mat.Specularity()))
    {
        // Set the shadow ray as specular ray, we overwrite the ray state
        // but ray state is important for rays that hit light.
        // And next call (material-related one) will overwrite it
        // anyway.
        dataPack.type = RayType::SPECULAR_RAY;
    }
    else
    {
        shadowMediaPack.SetPassthrough(shadowMatEval.isPassedThrough);
        rS.dShadowMediaListPack[rayIndex] = shadowMediaPack;

        Vector3 nudgeNormal = surf.geoNormal;
        //
        if(shadowMatEval.isPassedThrough) nudgeNormal *= Float(-1);
        //
        shadowRay = shadowRay.Nudge(nudgeNormal);
        RayToGMem(rS.dShadowRays, rayIndex, shadowRay, shadowTMM);
        RayCone rayConeOut = rConeRefract.ConeAfterScatter(shadowRay.dir,
                                                           surf.geoNormal);
        rS.dShadowRayCones[rayIndex]    = rayConeOut;
        rS.dShadowRayRadiance[rayIndex] = shadowRadiance;
        rS.dRPathPDFShadow[rayIndex]    = rPathShadow;
        rS.dRLightPDFShadow[rayIndex]   = rLightShadow;
    }

    // Generic Write
    rS.dPathDataPack[rayIndex] = dataPack;
}

// =============================== //
//       MEDIA LIGHT SURFACE       //
// =============================== //
template<LightGroupC L, TransformGroupC T, class SC, class LS>
MR_HF_DEF
void LightWorkFunctionMedia<L, T, SC, LS>::Call(const Light&, RNGDispenser&, const SpectrumConv&,
                                                const Params&, RayIndex, uint32_t)
{
    //if(params.globalState.sampleMedia)
    //    params.rayState.dMediaListPack[rayIndex].SetPassthrough(raySample.eval.isPassedThrough);


    //PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    //if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    //// If mode is NEE, only camera rays are allowed
    //if(params.globalState.sampleMode == SampleMode::E::NEE &&
    //   (pathDataPack.type != RayType::CAMERA_RAY) &&
    //   (pathDataPack.type != RayType::SPECULAR_RAY))
    //{
    //    pathDataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    //    params.rayState.dPathDataPack[rayIndex] = pathDataPack;
    //    return;
    //}

    //bool switchToMISPdf = (pathDataPack.type == RayType::PATH_RAY);
    //Spectrum throughput = params.rayState.dThroughput[rayIndex];
    //auto [ray, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    //RayCone rayCone = params.common.dRayCones[rayIndex].Advance(tMM[1]);
    //if(switchToMISPdf)
    //{
    //    using Distribution::MIS::BalanceCancelled;
    //    using Distribution::Common::DivideByPDF;
    //    //
    //    Array<Float, 2> weights = {Float(1), Float(1)};
    //    Array<Float, 2> pdfs;
    //    pdfs[0] = params.rayState.dPrevMatPDF[rayIndex];
    //    // We need to find the index of this specific light
    //    // Light sampler will handle it
    //    HitKeyPack hitKeyPack = params.common.dKeys[rayIndex];
    //    MetaHit hit = params.common.dHits[rayIndex];
    //    pdfs[1] = params.globalState.lightSampler.PdfLight(hitKeyPack, hit, ray);
    //    Float misPdf = BalanceCancelled<2>(pdfs, weights);
    //    // We premultiply the throughput under the assumption this will not hit a light,
    //    // but we did. So revert the multiplication first then multiply with
    //    // MIS weight.
    //    throughput *= pdfs[0];
    //    throughput = DivideByPDF(throughput, misPdf);
    //}

    //Vector3 wO = -ray.dir;
    //Spectrum emission;
    //if constexpr(Light::IsPrimitiveBackedLight)
    //{
    //    // It is more accurate to use hit if we actually hit the material
    //    using Hit = typename Light::Primitive::Hit;
    //    static constexpr uint32_t N = Hit::Dims;
    //    MetaHit metaHit = params.common.dHits[rayIndex];
    //    Hit hit = metaHit.AsVector<N>();
    //    emission = l.EmitViaHit(wO, hit, rayCone);
    //}
    //else
    //{
    //    Vector3 position = ray.AdvancedPos(tMM[1]);
    //    emission = l.EmitViaSurfacePoint(wO, position, rayCone);
    //}

    //// Check the depth if we exceed it, do not accumulate
    //// we terminate the path regardless
    //Vector2ui rrRange = params.globalState.russianRouletteRange;
    //if((pathDataPack.depth + 1u) <= rrRange[1])
    //{
    //    Spectrum radianceEstimate = emission * throughput;
    //    params.rayState.dPathRadiance[rayIndex] += radianceEstimate;
    //}
    //// Set the path as dead
    //pathDataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    //params.rayState.dPathDataPack[rayIndex] = pathDataPack;
}

// ========================= //
//      MEDIA KERNELS        //
// ========================= //
//   PURE PATH TRACE MEDIUM  //
// ========================= //
template<MediumGroupC M, TransformGroupC T, class SC>
MR_HF_DEF
void MediumWorkFunction <M, T, SC>::Call(const Medium& medium, const TContext& tContext, SpectrumConv&,
                                         RNGDispenser& rng, const Params& params, RayIndex rIndex, uint32_t)
{
    using namespace Distribution::Common;
    using enum MediumEvent;
    using MediumTraverser = typename Medium::Traverser;

    // RGB / Spectral agnostic MIS pdf calculation
    // TODO: This should be somewhere else in common or
    // attached to the Spectrum Context.
    //
    // Calculates the balance heuristic.
    static constexpr uint32_t COLOR_CHANNELS = (SpectrumConv::IsRGB)
                                                ? 3u
                                                : SpectraPerSpectrum;
    static constexpr Float COLOR_CHANNELS_INV = Float(1) / Float(COLOR_CHANNELS);
    auto ColorChannelMIS = [](Spectrum pdfs)
    {
        Float pdf = Float(0);
        MRAY_UNROLL_LOOP
        for(uint32_t i = 0; i < COLOR_CHANNELS; i++)
            pdf += pdfs[i];

        return pdf * COLOR_CHANNELS_INV;
    };

    const auto& rS          = params.rayState;
    const auto& gS          = params.globalState;
    auto [ray, tMM]         = RayFromGMem(params.common.dRays, rIndex);
    auto rngBackup          = BackupRNG(rS.dBackupRNGStates[rIndex]);
    PathDataPack pathState  = rS.dPathDataPack[rIndex];
    Spectrum throughput     = rS.dThroughput[rIndex];
    Spectrum pathRadiance   = rS.dPathRadiance[rIndex];
    const Vector2ui rrRange = gS.russianRouletteRange;

    // To medium-local ray
    Ray lRay = tContext.InvApply(ray);
    Float lLen = Math::Length(lRay.dir);
    Vector2 lTMM = tMM * lLen;
    lRay.dir *= (Float(1) / lLen);
    // Due to numerical imprecisions, this can be INF
    // we do not want that since media sampler uses INF
    // when maorant segment has the value of zero
    // (Exponential sample will return INF in that case)
    // we use that
    lTMM[1] = Math::Min(lTMM[1], std::numeric_limits<Float>::max());

    // MIS of Channel / Spectra
    uint32_t sampleChannelIndex = 0;
    if constexpr(SpectrumConv::IsRGB)
        sampleChannelIndex = uint32_t(rngBackup.NextFloat() * COLOR_CHANNELS);

    // Try to punchthrough the medium
    bool emissionAccumulated = false;
    MediumEvent status = TRANSMITTED;
    MediumTraverser mt = medium.GenTraverser(lRay, lTMM);
    Spectrum sMaj, tMaj; Float tRay = lTMM[0];
    while(mt.SampleTMajor(tMaj, sMaj, tRay, rngBackup.NextFloat(),
                          sampleChannelIndex))
    {
        Vector3 point = lRay.AdvancedPos(tRay);
        MediumQuery mQuery = medium.Query(point, rngBackup.NextFloat());
        // ============ //
        //   Emission   //
        // ============ //
        if(mQuery.emission && (pathState.depth + 1u) < rrRange[1])
        {
            emissionAccumulated = true;
            Spectrum emission = mQuery.emission.Value();

            Spectrum factor = tMaj * sMaj;
            Float pdfBalance = ColorChannelMIS(factor);
            factor = DivideByPDF(factor, pdfBalance);
            Spectrum tpEmit = throughput * factor * mQuery.sigmaA;
            pathRadiance += tpEmit * emission;
        }
        // Next Event
        Float wFactor = Float(1) / sMaj[sampleChannelIndex];
        Float probA = mQuery.sigmaA[sampleChannelIndex] * wFactor;
        Float probS = mQuery.sigmaS[sampleChannelIndex] * wFactor;
        Float probN = Float(1) - probA - probS;
        Array<Float, 3> dist = {probA, probS, probN};
        auto [statusInt, localXi] = BisectSample<3>(rngBackup.NextFloat(), dist, true);
        status = MediumEvent(statusInt);
        if(status == ABSORBED)
        {
            throughput = Spectrum::Zero();
            pathState.status.Set(uint32_t(PathStatusEnum::DEAD));
            break;
        }
        else if(status == SCATTERED)
        {
            pathState.status.Set(uint32_t(PathStatusEnum::MEDIUM_SCATTERED));
            pathState.depth++;
            auto sample = medium.SampleScattering(lRay.dir, point, rng);

            Spectrum factor = tMaj * mQuery.sigmaS;
            Float pdfBalance = ColorChannelMIS(factor);
            throughput *= DivideByPDF(factor, pdfBalance);
            throughput *= Spectrum(sample.value.phaseVal);
            throughput = DivideByPDF(throughput, sample.pdf);

            // ================ //
            // Russian Roulette //
            // ================ //
            bool isPathDead = (pathState.depth >= rrRange[1]);
            if(!isPathDead && pathState.depth >= rrRange[0])
            {
                // TODO: This is still wrong maybe? What about dispersion?
                Float rrXi = rng.NextFloat<Medium::SampleScatteringRNList.TotalRNCount()>();
                Float rrFactor = throughput.Sum() * COLOR_CHANNELS_INV;
                auto result = RussianRoulette(throughput, rrFactor, rrXi);
                isPathDead = !result.HasValue();
                throughput = result.ValueOr(throughput);

                if(isPathDead)
                    pathState.status.Set(uint32_t(PathStatusEnum::DEAD));
            }
            // Outputs
            Float tNext = tRay * lLen;
            // Set Ray
            Vector3 pointOut = lRay.AdvancedPos(tNext);
            ray = tContext.Apply(Ray(sample.value.wI, pointOut));
            // Set Cone
            RayCone rCone = params.common.dRayCones[rIndex].Advance(tNext);
            params.common.dRayCones[rIndex] = rCone;
            // Set TMM
            tMM = Vector2(0, std::numeric_limits<Float>::max());
            break;
        }
        // Null scattering
        Spectrum sigmaN = sMaj - mQuery.sigmaA - mQuery.sigmaS;
        Spectrum factor = tMaj * sigmaN;
        Float pdfBalance = ColorChannelMIS(factor);
        throughput *= DivideByPDF(factor, pdfBalance);

        if(throughput < Spectrum(MathConstants::SmallEpsilon<Float>()))
            break;
    }
    assert(status == TRANSMITTED || status == ABSORBED ||
           status == SCATTERED);

    if(status == TRANSMITTED)
    {
        pathState.status.Set(uint32_t(PathStatusEnum::MEDIUM_TRANSMITTED));



        Spectrum factor = DivideByPDF(tMaj, tMaj[0]);
        throughput *= factor;

    }
    if(emissionAccumulated)
        rS.dPathRadiance[rIndex] = pathRadiance;

    RayToGMem(params.common.dRays, rIndex, ray, tMM);
    rS.dThroughput[rIndex] = throughput;
    rS.dPathDataPack[rIndex] = pathState;
}

// ========================= //
// NEE/MIS PATH TRACE MEDIUM //
// ========================= //
template<MediumGroupC M, TransformGroupC T, class SC, class LS>
MR_HF_DEF
void MediumWorkFunctionWithNEE<M, T, SC, LS>::Call(const Medium& medium, const TContext& tContext,
                                                   SpectrumConv& specConverter, RNGDispenser& rng,
                                                   const Params& params,
                                                   RayIndex rIndex, uint32_t)
{
    using namespace Distribution::Common;
    using enum MediumEvent;
    using MediumTraverser = typename Medium::Traverser;
    using LightSampler = LS;

    const auto& rS          = params.rayState;
    const auto& gS          = params.globalState;
    auto [ray, tMM]         = RayFromGMem(params.common.dRays, rIndex);
    auto rngBackup          = BackupRNG(rS.dBackupRNGStates[rIndex]);
    Spectrum rLight         = rS.dRLightPDF[rIndex];
    Spectrum rPath          = rS.dRPathPDF[rIndex];
    PathDataPack pathState  = rS.dPathDataPack[rIndex];
    Spectrum throughput     = rS.dThroughput[rIndex];
    Spectrum pathRadiance   = rS.dPathRadiance[rIndex];
    const Vector2ui rrRange = gS.russianRouletteRange;

    // To medium-local ray
    Ray lRay = tContext.InvApply(ray);
    Float lLen = Math::Length(lRay.dir);
    Vector2 lTMM = tMM * lLen;
    lRay.dir *= (Float(1) / lLen);
    // Due to numerical imprecisions, this can be INF
    // we do not want that since media sampler uses INF
    // when maorant segment has the value of zero
    // (Exponential sample will return INF in that case)
    // we use that
    lTMM[1] = Math::Min(lTMM[1], std::numeric_limits<Float>::max());

    // Try to punchthrough the medium
    bool emissionAccumulated = false;
    MediumEvent status = TRANSMITTED;
    MediumTraverser mt = medium.GenTraverser(lRay, lTMM);
    Spectrum sMaj, tMaj; Float tRay = lTMM[0];
    while(mt.SampleTMajor(tMaj, sMaj, tRay, rngBackup.NextFloat()))
    {
        Vector3 point = lRay.AdvancedPos(tRay);
        MediumQuery mQuery = medium.Query(point, rngBackup.NextFloat());
        // ============ //
        //   Emission   //
        // ============ //
        if(mQuery.emission && (pathState.depth + 1u) < rrRange[1])
        {
            emissionAccumulated = true;
            Spectrum emission = mQuery.emission.Value();
            Float pdf = tMaj[0] * sMaj[0];
            Spectrum factor = DivideByPDF(tMaj, pdf);
            Spectrum tpEmit = throughput * factor;
            Spectrum rEmit = rPath * sMaj * factor;
            // TODO: This is wrong for RGB renderers
            //static constexpr auto N = SpectraPerSpectrum;
            Float rEMIS = rEmit.Sum();
            //
            Spectrum eOut = tpEmit * mQuery.sigmaA * emission;
            pathRadiance += DivideByPDF(eOut, rEMIS);
        }
        // Next Event
        Float wFactor = Float(1) / sMaj[0];
        Float probA = mQuery.sigmaA[0] * wFactor;
        Float probS = mQuery.sigmaS[0] * wFactor;
        Float probN = Float(1) - probA - probS;
        Array<Float, 3> dist = {probA, probS, probN};
        auto [statusInt, localXi] = BisectSample<3>(rngBackup.NextFloat(), dist, true);
        status = MediumEvent(statusInt);
        if(status == ABSORBED)
        {
            throughput = Spectrum::Zero();
            pathState.status.Set(uint32_t(PathStatusEnum::DEAD));
            break;
        }
        else if(status == SCATTERED)
        {
            // Local point
            Float tNext = tRay * lLen;
            Vector3 localPos = lRay.AdvancedPos(tNext);
            Vector3 scatterWorldPos = tContext.ApplyP(localPos);

            // Common PDF / Factor
            Float scatterPDF = tMaj[0] * mQuery.sigmaS[0];
            Spectrum scatterFactor = DivideByPDF(tMaj * mQuery.sigmaS, scatterPDF);
            throughput *= scatterFactor;

            // ================ //
            //       NEE        //
            // ================ //
            // Now we do not have surface information (normal)
            // for the statistical particle.
            // We need it for light sampler since light sampler
            // automatically bends the ray cone depending on the sampled side.
            //
            // Media particles do not have curvature so we create our own ray cone
            // surface here with zero beta (curvature).
            RayCone rayCone = params.common.dRayCones[rIndex].Advance(tNext);
            RayConeSurface particleSurfaceCone =
            {
                .rayConeFront = rayCone,
                .rayConeBack  = rayCone,
                .betaN        = Float(0)
            };
            const LightSampler& lightSampler = gS.lightSampler;
            LightSample lightSample = lightSampler.SampleLight(rng, specConverter,
                                                               scatterWorldPos,
                                                               Vector3::ZAxis(),
                                                               particleSurfaceCone);
            auto [shadowRay, shadowTMM] = lightSample.value.SampledRay(scatterWorldPos);
            Vector3 shadowWI = tContext.InvApplyV(shadowRay.dir);
            Float shadowPhasePDF = medium.PdfScattering(shadowWI, lRay.dir, point);
            Float shadowPhaseVal = medium.EvalScattering(shadowWI, lRay.dir, point);

            // Store the shadow ray and shadow ray pdf ratios
            // A scattering event has occured (in this case medium),
            // so pdfs are the same.
            rS.dRPathPDFShadow[rIndex] = rPath * scatterFactor;
            rS.dRLightPDFShadow[rIndex] = rPath * scatterFactor;
            //
            RayToGMem(rS.dShadowRays, rIndex, shadowRay, shadowTMM);
            rS.dShadowRayCones[rIndex] = rayCone;
            Spectrum shadowThroughput = DivideByPDF(throughput, lightSample.pdf);
            shadowThroughput *= DivideByPDF(shadowPhaseVal, shadowPhasePDF);
            // Since multiplication is associative, we pre-multiply with
            // emission etc.
            // Recursive shadow ray caster will shed this value if ray reaches
            // light.
            rS.dShadowRayRadiance[rIndex] = (shadowThroughput *
                                             lightSample.value.emission);

            // ================ //
            //  Scattered Ray   //
            // ================ //
            pathState.status.Set(uint32_t(PathStatusEnum::MEDIUM_SCATTERED));
            pathState.depth++;
            auto sample = medium.SampleScattering(lRay.dir, point, rng);
            throughput *= Spectrum(sample.value.phaseVal);
            throughput = DivideByPDF(throughput, sample.pdf);
            rPath *= scatterFactor;
            rLight = rPath;

            // ================ //
            // Russian Roulette //
            // ================ //
            bool isPathDead = (pathState.depth >= rrRange[1]);
            if(!isPathDead && pathState.depth >= rrRange[0])
            {
                // TODO: This is still wrong maybe? What about dispersion?
                static constexpr Float ChannelCountInv = (SpectrumConv::IsRGB)
                    ? Float(0.33333333)
                    : Float(1) / Float(SpectraPerSpectrum);

                Float rrXi = rng.NextFloat<Medium::SampleScatteringRNList.TotalRNCount()>();
                Float rrFactor = throughput.Sum() * ChannelCountInv;
                auto result = RussianRoulette(throughput, rrFactor, rrXi);
                isPathDead = !result.HasValue();
                throughput = result.ValueOr(throughput);

                if(isPathDead)
                    pathState.status.Set(uint32_t(PathStatusEnum::DEAD));
            }

            // Outputs
            // Set Ray
            Vector3 worldWI = tContext.ApplyV(sample.value.wI);
            ray = Ray(worldWI, scatterWorldPos);
            // Set Cone
            RayCone rCone = params.common.dRayCones[rIndex].Advance(tNext);
            params.common.dRayCones[rIndex] = rCone;
            // Set TMM
            tMM = Vector2(0, std::numeric_limits<Float>::max());

            break;
        }
        // Null scattering
        Spectrum sigmaN = sMaj - mQuery.sigmaA - mQuery.sigmaS;
        Float pdf = tMaj[0] * sigmaN[0];
        Spectrum factor = DivideByPDF(tMaj * sigmaN, pdf);
        throughput *= factor;

        rPath *= factor;
        rLight *= DivideByPDF(tMaj * sMaj, pdf);

        if(throughput < Spectrum(MathConstants::SmallEpsilon<Float>()))
            break;
    }
    assert(status == TRANSMITTED || status == ABSORBED ||
           status == SCATTERED);

    if(status == TRANSMITTED)
    {
        pathState.status.Set(uint32_t(PathStatusEnum::MEDIUM_TRANSMITTED));

        Spectrum factor = DivideByPDF(tMaj, tMaj[0]);
        throughput *= factor;
        rPath *= factor;
        rLight *= factor;
    }
    if(emissionAccumulated)
        rS.dPathRadiance[rIndex] = pathRadiance;

    RayToGMem(params.common.dRays, rIndex, ray, tMM);
    rS.dThroughput[rIndex] = throughput;
    rS.dRPathPDF[rIndex] = rPath;
    rS.dRLightPDF[rIndex] = rLight;
    rS.dPathDataPack[rIndex] = pathState;
}

// =============================== //
// PATH TRACE MEDIUM TRANSMITTANCE //
//     (For Shadow Ray Casting)    //
// =============================== //
template<MediumGroupC M, TransformGroupC T, class SC>
MR_HF_DEF
void MediumWorkFunctionTransmittance<M, T, SC>::Call(const Medium&, const TContext&, SpectrumConv&,
                                                     RNGDispenser&, const Params&, RayIndex, uint32_t)
{}

}
