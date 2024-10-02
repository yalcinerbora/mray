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
    GenericEnum(E eIn) : e(eIn) {}
    constexpr operator E() const { return e; }
    constexpr operator E() { return e; }
    //
    constexpr std::string_view ToString(E e)
    {
        assert(e < E::END);
        return Names[e];
    }

    constexpr E FromString(std::string_view sv)
    {
        auto loc = std::find_if(Names.cbegin(), Names.cend(),
                                [&](std::string_view r)
        {
            return sv == r;
        });
        assert(loc != Names.cend());
        return E(std::distance(Names.cbegin(), loc));
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

    enum class RayType : uint32_t
    {
        SHADOW_RAY,
        SPECULAR_PATH_RAY,
        PATH_RAY,
        CAMERA_RAY
    };

    struct Options
    {
        uint32_t            totalSPP = 32;
        Vector2ui           russionRouletteRange = Vector2ui(3, 21);
        LightSamplerType    lightSampler = LightSamplerType::E::UNIFORM;
        SampleMode          sampleMode = SampleMode::E::NEE_WITH_MIS;
    };

    template<class LightSampler>
    struct GlobalState
    {
        Vector2ui       russionRouletteRange;
        SampleMode      sampleMode;
        LightSampler    lightSampler;
    };

    struct RayState
    {
        // Output related
        Span<Spectrum>          dPathRadiance;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
        // Path state
        Span<Spectrum>          dThroughput;
        Span<RayType>           dPathRayType;
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

template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::WorkFunction(const Prim&, const Material&, const Surface& surf,
                                    const TContext& tContext, RNGDispenser&,
                                    const WorkParams<EmptyType, PG, MG, TG>& params,
                                    RayIndex rayIndex)
{

}

template<class LightSampler, PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
void PathTraceRDetail::WorkFunctionWithNEE(const Prim&, const Material& mat, const Surface& surf,
                                           const TContext& tContext, RNGDispenser& rng,
                                           const WorkParams<LightSampler, PG, MG, TG>& params,
                                           RayIndex rayIndex)
{
}

template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunction(const Light& l, RNGDispenser&,
                                         const LightWorkParams<EmptyType, LG, TG>& params,
                                         RayIndex rayIndex)
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

template<class LightSampler, LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunctionWithNEE(const Light& l, RNGDispenser&,
                                                const LightWorkParams<LightSampler, LG, TG>& params,
                                                RayIndex rayIndex)
{

}
