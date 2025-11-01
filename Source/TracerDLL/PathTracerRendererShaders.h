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
    };

    template<class LightSampler, class SpectrumConverter>
    struct GlobalState
    {
        using SpecConverterData = typename SpectrumConverter::Data;

        Vector2ui           russianRouletteRange;
        SampleMode          sampleMode;
        LightSampler        lightSampler;
        SpecConverterData   specContextData;
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
        // Next set of rays
        Span<RayGMem>           dOutRays;
        Span<RayCone>           dOutRayCones;
        // Only used when NEE/MIS is active
        Span<Float>             dPrevMatPDF;
        Span<Spectrum>          dShadowRayRadiance;
        // May be empty if not spectral
        Span<SpectrumWaves>     dPathWavelengths;
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

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType,
             class SpectrumCtxType>
    struct WorkFunction
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType, SpectrumCtxType, 1u);
        using Params = WorkParams<EmptyType, SpectrumConv, PG, MG, TG>;

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
        using Params = WorkParams<LightSampler, SpectrumConv, PG, MG, TG>;

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
        using Params = LightWorkParams<EmptyType, SpectrumConv, LG, TG>;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params&, RayIndex rayIndex, uint32_t laneId);

    };

    template<LightGroupC LGType, TransformGroupC TGType,
             class SpectrumCtxType, class LightSampler>
    struct LightWorkFunctionWithNEE
    {
        MRAY_LIGHT_WORK_FUNCTOR_DEFINE_TYPES(LGType, TGType, SpectrumCtxType, 1u);
        using Params = LightWorkParams<LightSampler, SpectrumConv, LG, TG>;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&, const SpectrumConv&,
                         const Params& params, RayIndex rayIndex, uint32_t laneId);
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
    Vector3 wO = Math::Normalize(tContext.InvApplyN(-rayIn.dir));
    RayConeSurface rConeRefract = mat.RefractRayCone(surfRayCone, wO);
    BxDFSample raySample = mat.SampleBxDF(wO, rng);
    raySample.wI.dir = Math::Normalize(tContext.ApplyN(raySample.wI.dir));

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
        Float rrXi = rng.NextFloat<Material::SampleRNList.TotalRNCount()>();
        Float rrFactor = throughput.Sum() * Float(0.33333);
        auto result = RussianRoulette(throughput, rrFactor, rrXi);
        isPathDead = !result.has_value();
        throughput = result.value_or(throughput);
    }

    // Change the ray type, if mat is highly specular
    // we do not bother casting NEE ray. So if this ray
    // hits a light somehow it should not assume MIS is enabled.
    bool isSpecular = MaterialCommon::IsSpecular(specularity);
    dataPack.type = isSpecular ? RayType::SPECULAR_RAY : RayType::PATH_RAY;

    // Selectively write if path is alive
    if(!isPathDead)
    {
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
        RayToGMem(params.rayState.dOutRays, rayIndex, rayOut, tMMOut);
        // Continue the ray cone
        params.rayState.dOutRayCones[rayIndex] = rayConeOut;
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

    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    // ================ //
    //       NEE        //
    // ================ //
    auto [rayIn, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    Vector3 wO = Math::Normalize(-tContext.InvApplyN(rayIn.dir));
    const LightSampler& lightSampler = params.globalState.lightSampler;
    Vector3 worldPos = surf.position;
    RayConeSurface refractedRayCone = mat.RefractRayCone(surfRayCone, wO);
    LightSample lightSample = lightSampler.SampleLight(rng, specConverter,
                                                       worldPos, surf.geoNormal,
                                                       refractedRayCone);
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
        Float bxdfPdf = mat.Pdf(shadowRay, wO);
        std::array<Float, 2> pdfs = {bxdfPdf, lightSample.pdf};
        std::array<Float, 2> weights = {1, 1};
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
        RayToGMem(params.rayState.dOutRays, rayIndex,
                  shadowRay, shadowTMM);
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
        std::array<Float, 2> weights = {1, 1};
        std::array<Float, 2> pdfs;
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

}