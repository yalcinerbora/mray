#pragma once

#include <array>
#include <string_view>

#include "Core/ColorFunctions.h"

#include "PrimitiveC.h"
#include "MaterialC.h"
#include "RenderWork.h"
#include "DistributionFunctions.h"

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

    struct Options
    {
        uint32_t            totalSPP = 32;
        Vector2ui           russionRouletteRange = Vector2ui(3, 21);
        LightSamplerType    lightSampler = LightSamplerType::E::UNIFORM;
        SampleMode          sampleMode;
    };

    struct GlobalState
    {
        Vector2ui   russionRouletteRange;
    };

    struct RayState
    {
        // Can be position, furance radiance, normal
        // or a false color
        Span<Spectrum>          dOutputData;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
    };
    // No payload (this is incident renderer so
    // everything is on ray state)
    using RayPayload = EmptyType;

    template<PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
        PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
        void WorkFunction(const Prim&, const Material&, const Surface&,
                          const TContext&, RNGDispenser&,
                          const RenderWorkParams<PathTracerRenderer, PG, MG, TG>& params,
                          RayIndex rayIndex);

    template<PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
        PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
        void WorkFunctionWithNEE(const Prim&, const Material&, const Surface&,
                                 const TContext&, RNGDispenser&,
                                 const RenderWorkParams<PathTracerRenderer, PG, MG, TG>& params,
                                 RayIndex rayIndex);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
        void LightWorkFunction(const Light&, RNGDispenser&,
                               const RenderLightWorkParams<PathTracerRenderer, LG, TG>& params,
                               RayIndex rayIndex);

    MRAY_HYBRID
        void InitRayState(const RayPayload&, const RayState&,
                          const RaySample&, uint32_t writeIndex);
}

template<PrimitiveC Prim, MaterialC Material,
    class Surface, class TContext,
    PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::WorkFunction(const Prim&, const Material&, const Surface& surf,
                                    const TContext& tContext, RNGDispenser&,
                                    const RenderWorkParams<PathTracerRenderer, PG, MG, TG>& params,
                                    RayIndex rayIndex)
{

}

template<PrimitiveC Prim, MaterialC Material,
    class Surface, class TContext,
    PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
void PathTraceRDetail::WorkFunctionWithNEE(const Prim&, const Material& mat, const Surface& surf,
                                           const TContext& tContext, RNGDispenser& rng,
                                           const RenderWorkParams<PathTracerRenderer, PG, MG, TG>& params,
                                           RayIndex rayIndex)
{
}

template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void PathTraceRDetail::LightWorkFunction(const Light& l, RNGDispenser&,
                                         const RenderLightWorkParams<PathTracerRenderer, LG, TG>& params,
                                         RayIndex rayIndex)
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
void PathTraceRDetail::InitRayState(const RayPayload&,
                                    const RayState& dStates,
                                    const RaySample& raySample,
                                    uint32_t writeIndex)
{
    dStates.dImageCoordinates[writeIndex] = raySample.value.imgCoords;
    dStates.dFilmFilterWeights[writeIndex] = raySample.pdf;
}
