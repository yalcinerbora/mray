#pragma once

#include <array>
#include <string_view>

#include "Core/ColorFunctions.h"

#include "PrimitiveC.h"
#include "MaterialC.h"
#include "RenderWork.h"
#include "DistributionFunctions.h"
#include "LightSampler.h"

template<class ML>
class PathTracerRenderer;

// Lets use this on path tracer, then we make it over the CoreLib
// so we can refine as needed
template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
class GenericEnum
{
    static constexpr std::array Names = NamesIn;

    public:
    using E = Enum;
    // This do not work :(
    //using enum E;

    private:
    E e;

    public:
    GenericEnum() = default;
    constexpr GenericEnum(E eIn) : e(eIn) {}
    constexpr GenericEnum(std::string_view sv)
    {
        auto loc = std::find_if(Names.cbegin(), Names.cend(),
                                [&](std::string_view r)
        {
            return sv == r;
        });
        if(loc == Names.cend())
            throw MRayError("Bad enum name");

        e = E(std::distance(Names.cbegin(), loc));
    }


    constexpr operator E() const { return e; }
    constexpr operator E() { return e; }
    //
    constexpr std::string_view ToString() const
    {
        assert(e < E::END);
        return Names[static_cast<uint32_t>(e)];
    }
};

namespace PathTraceRDetail
{
    enum class LightSamplerEnum
    {
        UNIFORM,
        IRRAD_WEIGHTED,
        //
        END
    };
    static constexpr std::array LightSamplerNames =
    {
        "Uniform",
        "IrradianceWeighted"
    };
    using LightSamplerType = GenericEnum<LightSamplerEnum, LightSamplerNames>;

    enum class SampleModeEnum
    {
        PURE,
        NEE,
        NEE_WITH_MIS,
        //
        END
    };
    static constexpr std::array SampleModeNames =
    {
        "Pure",
        "WithNextEventEstimation",
        "WithNEEAndMIS"
    };
    using SampleMode = GenericEnum<SampleModeEnum, SampleModeNames>;

    enum class RenderModeEnum
    {
        THROUGHPUT,
        LATENCY,
        //
        END
    };
    static constexpr std::array RenderModeNames =
    {
        "Throughput",
        "Latency"
    };
    using RenderMode = GenericEnum<RenderModeEnum, RenderModeNames>;

    enum class RayType : uint8_t
    {
        SHADOW_RAY,
        SPECULAR_RAY,
        PATH_RAY,
        CAMERA_RAY
    };

    enum class PathStatusEnum : uint8_t
    {
        // Path is dead (due to russian roulette or hitting a light source)
        DEAD                = 0,
        // Invalid rays are slightly different, sometimes due to exactly
        // meeting the spp requirement, renderer may not launch rays,
        // ıt will mark these as invalid in the pool
        INVALID             = 1,
        // TODO: These are not used yet, but here for future use.
        // Path did scatter because of the medium. It should not go
        // Material scattering
        MEDIUM_SCATTERED    = 2,
        // TODO: Maybe incorporate ray type?
        //
        END
    };
    using PathStatus = Bitset<static_cast<size_t>(PathStatusEnum::END)>;

    struct Options
    {
        uint32_t            totalSPP = 32;
        Vector2ui           russianRouletteRange = Vector2ui(4, 20);
        LightSamplerType    lightSampler = LightSamplerType::E::UNIFORM;
        SampleMode          sampleMode = SampleMode::E::PURE;
        RenderMode          renderMode = RenderMode::E::THROUGHPUT;
    };

    template<class LightSampler>
    struct GlobalState
    {
        Vector2ui       russianRouletteRange;
        SampleMode      sampleMode;
        LightSampler    lightSampler;
    };

    struct alignas(4) PathDataPack
    {
        uint8_t     depth;
        PathStatus  status;
        RayType     type;
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
        Span<RayDiff>           dOutRayDiffs;
        // Only used when NEE/MIS is active
        Span<Float>             dPrevMatPDF;
        Span<Spectrum>          dShadowRayRadiance;
        //Span<uint32_t>          dShadowRayLightIndex;
    };

    template<class LightSampler, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkParams = RenderWorkParams
    <
        GlobalState<LightSampler>,
        RayState,
        PG, MG, TG
    >;
    template<class LightSampler, LightGroupC LG, TransformGroupC TG>
    using LightWorkParams = RenderLightWorkParams
    <
        GlobalState<LightSampler>,
        RayState,
        LG, TG
    >;

    template<PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void WorkFunction(const Prim&, const Material&, const Surface&,
                      const TContext&, RNGDispenser&,
                      const WorkParams<EmptyType, PG, MG, TG>& params,
                      RayIndex rayIndex);

    template<class LightSampler,
             PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void WorkFunctionNEE(const Prim&, const Material&, const Surface&,
                         const TContext&, RNGDispenser&,
                         const WorkParams<LightSampler, PG, MG, TG>& params,
                         RayIndex rayIndex);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void LightWorkFunction(const Light&, RNGDispenser&,
                           const LightWorkParams<EmptyType, LG, TG>& params,
                           RayIndex rayIndex);
    template<class LightSampler, LightC Light,
             LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void LightWorkFunctionWithNEE(const Light&, RNGDispenser&,
                                  const LightWorkParams<LightSampler, LG, TG>& params,
                                  RayIndex rayIndex);
}

// ======================== //
//     PURE PATH TRACE      //
// ======================== //
template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::WorkFunction(const Prim&, const Material& mat, const Surface& surf,
                                    const TContext& tContext, RNGDispenser& rng,
                                    const WorkParams<EmptyType, PG, MG, TG>& params,
                                    RayIndex rayIndex)
{
    PathDataPack dataPack = params.rayState.dPathDataPack[rayIndex];
    if(dataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    // ================ //
    // Sample Material  //
    // ================ //
    auto [rayIn, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    Vector3 wO = tContext.InvApplyN(-rayIn.Dir()).Normalize();
    SampleT<BxDFResult> raySample = mat.SampleBxDF(wO, surf, rng);
    Vector3 wI = tContext.ApplyN(raySample.value.wI.Dir()).Normalize();
    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    throughput *= raySample.value.reflectance;

    // ================ //
    // Russian Roulette //
    // ================ //
    Float specularity = mat.Specularity(surf);
    dataPack.depth += 1u;
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    bool isPathDead = (dataPack.depth >= rrRange[1]);
    if(!isPathDead && dataPack.depth >= rrRange[0])
    {
        Float rrXi = rng.NextFloat<Material::SampleRNCount>();
        Float reflectanceAvg = raySample.value.reflectance.Sum() * Float(0.33333);
        Float throughputAvg = raySample.value.reflectance.Sum() * Float(0.33333);
        Float rrFactor = Math::Lerp(reflectanceAvg, throughputAvg, specularity);
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
        // If alive update througput
        params.rayState.dThroughput[rayIndex] = DivideByPDF(throughput, raySample.pdf);

        // Save the previous pdf (aka. current pdf, naming is for the user)
        // To correctly do MIS we will need it
        if(params.globalState.sampleMode == SampleMode::E::NEE_WITH_MIS)
            params.rayState.dPrevMatPDF[rayIndex] = raySample.pdf;

        // ================ //
        //  Scattered Ray   //
        // ================ //
        // TODO: Report bug on NVIDIA, Below code crashes on 1080 (CC_61)
        // ACCESS_VIOLATION on ptxas. But not on 3070ti mobile (CC_89)
        // ==========================================
        //Vector3 nudgeNormal = (raySample.value.isPassedThrough)
        //    ? -surf.geoNormal
        //    : surf.geoNormal;
        //Ray rayOut = Ray(wI, surf.position);
        // rayOut.NudgeSelf(nudgeNormal);
        // ==========================================
        Vector3 nudgeNormal = surf.geoNormal;
        if(raySample.value.isPassedThrough)
            nudgeNormal *= Float(-1);
        Ray rayOut = Ray(wI, surf.position);
        rayOut.NudgeSelf(nudgeNormal);
        // If I remember correctly, OptiX does not like INF on rays,
        // so we put flt_max here.
        Vector2 tMMOut = Vector2(0, std::numeric_limits<Float>::max());
        RayToGMem(params.rayState.dOutRays, rayIndex, rayOut, tMMOut);
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
template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunction(const Light& l, RNGDispenser&,
                                         const LightWorkParams<EmptyType, LG, TG>& params,
                                         RayIndex rayIndex)
{
    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    auto [ray, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    Vector3 wO = -ray.Dir();
    Spectrum emission;
    if constexpr(Light::IsPrimitiveBackedLight)
    {
        // It is more accurate to use hit if we actually hit the material
        using Hit = typename Light::Primitive::Hit;
        static constexpr uint32_t N = Hit::Dims;
        MetaHit metaHit = params.in.dHits[rayIndex];
        Hit hit = metaHit.AsVector<N>();
        emission = l.EmitViaHit(wO, hit);
    }
    else
    {
        Vector3 position = ray.AdvancedPos(tMM[1]);
        emission = l.EmitViaSurfacePoint(wO, position);
    }

    // Check the depth if we exceed it, do not accumulate
    // we terminate the path regardless
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

// ======================== //
//    NEE AND/OR MIS EXT    //
// ======================== //
template<class LightSampler, PrimitiveC Prim, MaterialC Material,
    class Surface, class TContext,
    PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
void PathTraceRDetail::WorkFunctionNEE(const Prim&, const Material& mat, const Surface& surf,
                                       const TContext& tContext, RNGDispenser& rng,
                                       const WorkParams<LightSampler, PG, MG, TG>& params,
                                       RayIndex rayIndex)
{
    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    if(pathDataPack.status[uint32_t(PathStatusEnum::INVALID)]) return;

    // TODO: We need to get the context from somewhere
    // &&
    // TODO: Add spectrum related stuff, this should not be
    // default constructed
    typename SpectrumConverterContextIdentity::Converter specConverter;
    using IdentityST = SpectrumConverterContextIdentity;

    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    // ================ //
    //       NEE        //
    // ================ //
    auto [rayIn, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    const LightSampler& lightSampler = params.globalState.lightSampler;
    Vector3 worldPos = surf.position;
    LightSample lightSample = lightSampler.SampleLight(rng, specConverter, worldPos);
    auto [shadowRay, shadowTMM] = lightSample.value.SampledRay(worldPos);
    //
    Vector3 wO = -tContext.InvApplyN(rayIn.Dir()).Normalize();
    Ray wI = Ray(tContext.InvApplyN(shadowRay.Dir()).Normalize(),
                 shadowRay.Pos());
    Spectrum reflectance = mat.Evaluate(wI, wO, surf);
    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    throughput *= reflectance;

    // Either do MIS or normal sampling
    Float pdf;
    if(params.globalState.sampleMode == SampleMode::E::NEE_WITH_MIS)
    {
        using Distribution::MIS::BalanceCancelled;
        Float bxdfPdf = mat.Pdf(shadowRay, wO, surf);
        std::array<Float, 2> pdfs = {bxdfPdf, lightSample.pdf};
        std::array<Float, 2> weights = {1, 1};
        pdf = BalanceCancelled<2>(pdfs, weights);
    }
    else pdf = lightSample.pdf;
    // Pre-calculate the result, we will only do a visibility
    // check and if it succeeds we accumulate later
    Spectrum shadowRadiance = throughput * lightSample.value.emission;
    shadowRadiance = DivideByPDF(shadowRadiance, pdf);

    // Writing
    if(MaterialCommon::IsSpecular(mat.Specularity(surf)))
    {
        // Set the shadow ray as specular ray, we overwrite the ray state
        // but ray state is important for rays that hit light.
        // And next call (material-related one) will overwrite it
        // anyway.
        pathDataPack.type = RayType::SPECULAR_RAY;
    }
    else
    {
        shadowRay.NudgeSelf(surf.geoNormal);
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
template<class LightSampler, LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunctionWithNEE(const Light& l, RNGDispenser&,
                                                const LightWorkParams<LightSampler, LG, TG>& params,
                                                RayIndex rayIndex)
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
    auto [ray, tMM] = RayFromGMem(params.in.dRays, rayIndex);
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

    Vector3 wO = -ray.Dir();
    Spectrum emission;
    if constexpr(Light::IsPrimitiveBackedLight)
    {
        // It is more accurate to use hit if we actually hit the material
        using Hit = typename Light::Primitive::Hit;
        static constexpr uint32_t N = Hit::Dims;
        MetaHit metaHit = params.in.dHits[rayIndex];
        Hit hit = metaHit.AsVector<N>();
        emission = l.EmitViaHit(wO, hit);
    }
    else
    {
        Vector3 position = ray.AdvancedPos(tMM[1]);
        emission = l.EmitViaSurfacePoint(wO, position);
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
