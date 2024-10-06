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

    enum class RayType : uint8_t
    {
        SHADOW_RAY,
        SPECULAR_PATH_RAY,
        PATH_RAY,
        CAMERA_RAY
    };

    enum class PathStatusEnum : uint8_t
    {
        // Path is dead (due to russian roulette or hitting a light source)
        DEAD                = 0,
        // TODO: These are not used yet, but here for future use.
        // Path did scatter because of the medium. It should not go
        // Material scattering
        MEDIUM_SCATTERED    = 1,
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
        uint8_t    depth;
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
        Span<Float>             dPrevMaterialPDF;
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
    void WorkFunctionWithNEE(const Prim&, const Material&, const Surface&,
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
    using Distribution::Common::RussianRoulette;
    using Distribution::Common::DivideByPDF;
    //
    auto [rayIn, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    Vector3 wO = -tContext.InvApplyN(rayIn.Dir()).Normalize();
    SampleT<BxDFResult> raySample = mat.SampleBxDF(wO, surf, rng);
    Vector3 wI = tContext.ApplyN(raySample.value.wI.Dir());
    Spectrum throughput = params.rayState.dThroughput[rayIndex];
    throughput *= raySample.value.reflectance;

    // ================ //
    // Russian Roulette //
    // ================ //
    bool isPathDead = false;
    PathDataPack dataPack = params.rayState.dPathDataPack[rayIndex];
    dataPack.depth += 1u;
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    if(dataPack.depth > rrRange[1])
    {
        isPathDead = true;
    }
    else if(dataPack.depth >= rrRange[0])
    {
        Float rrXi = rng.NextFloat<Material::SampleRNCount>();
        Float rrFactor = throughput.Sum() * Float(0.33333);
        auto result = RussianRoulette(throughput, rrFactor, rrXi);
        isPathDead = !result.has_value();
        throughput = result.value_or(throughput);
    }
    if(isPathDead) dataPack.status.Set(uint32_t(PathStatusEnum::DEAD));

    // Write the updated state back
    params.rayState.dPathDataPack[rayIndex] = dataPack;
    params.rayState.dThroughput[rayIndex] = DivideByPDF(throughput, raySample.pdf);

    // New ray
    Vector3 nudgeNormal = (raySample.value.isPassedThrough)
                            ? -surf.geoNormal
                            : surf.geoNormal;
    Ray rayOut = Ray(wI, surf.position);
    rayOut.NudgeSelf(nudgeNormal);
    // If I remember correctly, OptiX does not like INF on rays,
    // so we put flt_max here.
    Vector2 tMMOut = Vector2(0, std::numeric_limits<Float>::max());
    RayToGMem(params.rayState.dOutRays, rayIndex, rayOut, tMMOut);
}

template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunction(const Light& l, RNGDispenser&,
                                         const LightWorkParams<EmptyType, LG, TG>& params,
                                         RayIndex rayIndex)
{

    auto [ray, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    Spectrum throughput = params.rayState.dThroughput[rayIndex];

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
    PathDataPack pathDataPack = params.rayState.dPathDataPack[rayIndex];
    Vector2ui rrRange = params.globalState.russianRouletteRange;
    if((pathDataPack.depth + 1u) <= rrRange[1])
    {
        Spectrum radianceEstimate = emission * throughput;
        params.rayState.dPathRadiance[rayIndex] = radianceEstimate;
    }
    // Set the path as dead
    pathDataPack.status.Set(uint32_t(PathStatusEnum::DEAD));
    params.rayState.dPathDataPack[rayIndex] = pathDataPack;
}

// ======================== //
//      NEE AND/OR MIS      //
// ======================== //
template<class LightSampler, PrimitiveC Prim, MaterialC Material,
    class Surface, class TContext,
    PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
void PathTraceRDetail::WorkFunctionWithNEE(const Prim&, const Material&, const Surface&,
                                           const TContext&, RNGDispenser&,
                                           const WorkParams<LightSampler, PG, MG, TG>&,
                                           RayIndex)
{
}

template<class LightSampler, LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunctionWithNEE(const Light&, RNGDispenser&,
                                                const LightWorkParams<LightSampler, LG, TG>&,
                                                RayIndex)
{

    //using enum RayType;
    //using enum SampleMode::E;
    ////
    //RayType rType = params.rayState.dPathRayType[rayIndex];
    //const GlobalState& gState = params.globalState;

    //// If we call work function
    //// Only accumulate if
    //bool isSpecPathRay = (rType == SPECULAR_PATH_RAY);
    //bool isPathRay = (rType == PATH_RAY);
    //bool isPathRayAsMISRay = (gState.sampleMode == NEE_WITH_MIS && rType == PATH_RAY);
    //bool isCameraRay = (rType == CAMERA_RAY);
    //bool isSpecPathRay = (rType == SPECULAR_PATH_RAY);
    //bool isNEEOff = (gState.sampleMode == PURE);
    //bool doAccumulate = (isPathRayAsMISRay || isCameraRay || isSpecPathRay || isNEEOff);


    //// If we are doing direct light MIS and hit a light,
    //// actual path ray automatically becomes MIS ray.
    //Float misWeight = Float(1);
    //if(isPathRayAsMISRay)
    //{
    //    Float lightSamplePDF;
    //    // Find out the pdf of the light
    //    Float pdfLightM, pdfLightC;
    //    //renderState.gLightSampler->Pdf(...);

    //    // We are sub-sampling (discretely sampling) a single light
    //    // pdf of BxDF should also incorporate this
    //    Float bxdfPDF = params.rayState.dPrevMaterialPDF[rayIndex];
    //    misWeight = Distribution::MIS::Power(1, bxdfPDF, 1, lightSamplePDF);
    //}


    ////bool isNEEOn = (gState.sampleMode == NEE_WITH_MIS ||
    ////                gState.sampleMode == NEE);
    ////bool isNEERay = isNEEOn && rType == NEE_RAY;

    ////bool isPathRayNEEOff = ((!isNEEOn) &&
    ////                        (rType == PATH_RAY ||
    ////                         rType == SPECULAR_PATH_RAY));

    //auto [ray, tMM] = RayFromGMem(params.in.dRays, rayIndex);
    //Vector3 position;
    //if constexpr(Light::IsPrimitiveBackedLight)
    //{
    //    params.in.position;
    //    l.
    //}
    //else
    //{

    //}
    //// = surface.WorldPosition();

    //Float misWeight = Float(1);
    //if(isPathRayAsMISRay)
    //{
    //    Float lightSamplePDF;
    //    //// Find out the pdf of the light
    //    //Float pdfLightM, pdfLightC;
    //    //renderState.gLightSampler->Pdf(pdfLightM, pdfLightC,
    //    //                               //
    //    //                               gLight.GlobalLightIndex(),
    //    //                               ray.tMax,
    //    //                               position,
    //    //                               direction,
    //    //                               surface.worldToTangent);

    //    // We are sub-sampling (discretely sampling) a single light
    //    // pdf of BxDF should also incorporate this
    //    Float bxdfPDF = params.rayState.dShadowRayMaterialPDF[rayIndex];
    //    misWeight = Distribution::MIS::Power(1, bxdfPDF, 1, lightSamplePDF);
    //}

    //if(isPathRayNEEOff   || // We hit a light with a path ray while NEE is off
    //   isPathRayAsMISRay || // We hit a light with a path ray while MIS option is enabled
    //   isCorrectNEERay   || // We hit the correct light as a NEE ray while NEE is on
    //   isCameraRay       || // We hit as a camera ray which should not be culled when NEE is on
    //   isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should contribute
    //{

    //    Spectrum throughput = params.rayState.dPathThroughput[rIndex];
    //    Spectrum emission = l.EmitViaSurfacePoint(-ray.Dir(), position);
    //    Spectrum radianceEstimate =  emission * throughput;
    //    //
    //    radianceEstimate *= misWeight;
    //    //
    //    params.rayState.dPathRadiance[rayIndex] += radianceEstimate;
    //}
}
