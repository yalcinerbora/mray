#pragma once

#include <array>
#include <string_view>

#include "Core/ColorFunctions.h"

#include "RenderWork.h"
#include "DistributionFunctions.h"

class SurfaceRenderer;

namespace SurfRDetail
{

    struct Mode
    {
        public:
        enum E
        {
            AO,
            FURNACE,
            WORLD_NORMAL,
            WORLD_POSITION,
            WORLD_GEO_NORMAL,
            HIT_PARAMS,
            MAT_ID,
            PRIM_ID,
            ACCEL_ID,
            TRANSFORM_ID,
            UV,
            //
            END
        };

        private:
        static constexpr std::array Names =
        {
            "AmbientOcclusion",
            "Furnace",
            "WorldNormal",
            "WorldPosition",
            "WorldGeoNormal",
            "HitParams",
            "MaterialId",
            "PrimitiveId",
            "AcceleratorId",
            "TransformId",
            "UV"
        };
        static_assert(Names.size() == static_cast<size_t>(END),
                      "Not enough data on enum lookup table");

        public:
        E e;
        //
        static constexpr std::string_view ToString(E e)
        {
            assert(e < END);
            return Names[e];
        }

        static constexpr E FromString(std::string_view sv)
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

    struct Options
    {
        uint32_t    totalSPP            = 32;
        Mode::E     mode                = Mode::WORLD_NORMAL;
        bool        doStochasticFilter  = true;
        Float       tMaxAORatio         = Float(0.15);
    };

    struct GlobalState
    {
        // What are we rendering
        Mode mode;
        // For AO Renderer, secondary ray's tMax
        Float tMaxAO;
    };

    struct RayStateCommon
    {
        // Can be position, furnace radiance, normal
        // or a false color
        Span<Spectrum>          dOutputData;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
    };
    struct RayStateAO
    {
        // Can be position, furnace radiance, normal
        // or a false color
        Span<Spectrum>          dOutputData;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
        Span<RayGMem>           dVisibilityRays;
    };

    template<PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MR_HF_DECL
    void WorkFunctionCommon(const Prim&, const Material&, const Surface&,
                            const RayConeSurface&, const TContext&, RNGDispenser&,
                            const RenderWorkParams<GlobalState, RayStateCommon, PG, MG, TG>& params,
                            RayIndex rayIndex);

    template<PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MR_HF_DECL
    void WorkFunctionFurnaceOrAO(const Prim&, const Material&, const Surface&,
                                 const RayConeSurface&, const TContext&, RNGDispenser&,
                                 const RenderWorkParams<GlobalState, RayStateAO, PG, MG, TG>& params,
                                 RayIndex rayIndex);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MR_HF_DECL
    void LightWorkFunctionCommon(const Light&, RNGDispenser&,
                                 const RenderLightWorkParams<GlobalState, RayStateCommon, LG, TG>& params,
                                 RayIndex rayIndex);

}

template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MR_HF_DEF
void SurfRDetail::WorkFunctionCommon(const Prim&, const Material&, const Surface& surf,
                                     const RayConeSurface&, const TContext& tContext, RNGDispenser&,
                                     const RenderWorkParams<GlobalState, RayStateCommon, PG, MG, TG>& params,
                                     RayIndex rayIndex)
{
    Vector3 color = Vector3::Zero();
    Mode::E mode = params.globalState.mode.e;
    switch(mode)
    {
        using enum Mode::E;
        case WORLD_NORMAL:
        {
            if constexpr(std::is_same_v<BasicSurface, Surface>)
            {
                Vector3 normal = surf.normal;
                normal = (normal + Vector3(1)) * Vector3(0.5);
                color = normal;
            }
            else
            {
                Vector3 normal = surf.shadingTBN.OrthoBasisZ();
                normal = Math::Normalize(tContext.ApplyN(normal));
                normal = (normal + Vector3(1)) * Vector3(0.5);
                color = normal;
            }
            break;
        }
        case WORLD_GEO_NORMAL:
        {
            if constexpr(std::is_same_v<BasicSurface, Surface>)
            {
                Vector3 normal = surf.normal;
                normal = (normal + Vector3(1)) * Vector3(0.5);
                color = normal;
            }
            else
            {
                Vector3 normal = surf.geoNormal;
                normal = (normal + Vector3(1)) * Vector3(0.5);
                color = normal;
            }
            break;
        }
        case WORLD_POSITION:
        {
            color = surf.position;
            break;
        }
        case HIT_PARAMS:
        {
            MetaHit metaHit = params.in.dHits[rayIndex];
            Vector3 hit = Vector3(metaHit.AsVector<2u>(), Float(0));
            // We need to check the type here for triangles
            if constexpr(TrianglePrimGroupC<PG>)
            {
                hit[2] = Float(1) - hit[0] - hit[1];
            }
            color = hit;
            break;
        }
        case MAT_ID:
        {
            LightOrMatKey lmKey = params.in.dKeys[rayIndex].lightOrMatKey;
            MaterialKey mKey = MaterialKey::CombinedKey(lmKey.FetchBatchPortion(),
                                                        lmKey.FetchIndexPortion());
            CommonKey k = mKey.FetchBatchPortion() ^ mKey.FetchIndexPortion();
            color = Color::RandomColorRGB(static_cast<uint32_t>(k));
            break;
        }
        case PRIM_ID:
        {
            PrimitiveKey pKey = params.in.dKeys[rayIndex].primKey;
            CommonKey k = pKey.FetchBatchPortion() ^ pKey.FetchIndexPortion();
            color = Color::RandomColorRGB(static_cast<uint32_t>(k));
            break;
        }
        case ACCEL_ID:
        {
            AcceleratorKey aKey = params.in.dKeys[rayIndex].accelKey;
            CommonKey k = aKey.FetchBatchPortion() ^ aKey.FetchIndexPortion();
            color = Color::RandomColorRGB(static_cast<uint32_t>(k));
            break;
        }
        case TRANSFORM_ID:
        {
            TransformKey tKey = params.in.dKeys[rayIndex].transKey;
            CommonKey k = tKey.FetchBatchPortion() ^ tKey.FetchIndexPortion();
            color = Color::RandomColorRGB(static_cast<uint32_t>(k));
            break;
        }
        case UV:
        {
            if constexpr(std::is_same_v<DefaultSurface, Surface>)
            {
                color = Vector3(surf.uv, Float(0));
            }
            break;
        }
        default: color = BIG_MAGENTA(); break;
    }
    params.rayState.dOutputData[rayIndex] = Spectrum(color, Float(0));
}

template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MR_HF_DEF
void SurfRDetail::WorkFunctionFurnaceOrAO(const Prim&, const Material& mat, const Surface& surf,
                                          const RayConeSurface&, const TContext& tContext, RNGDispenser& rng,
                                          const RenderWorkParams<GlobalState, RayStateAO, PG, MG, TG>& params,
                                          RayIndex rayIndex)
{
    assert(params.globalState.mode.e == Mode::AO ||
           params.globalState.mode.e == Mode::FURNACE);

    if(params.globalState.mode.e == Mode::FURNACE)
    {
        using Distribution::Common::DivideByPDF;
        auto [rayIn, tMM] = RayFromGMem(params.in.dRays, rayIndex);
        Vector3 wO = rayIn.Dir();
        wO = Math::Normalize(tContext.InvApplyN(wO));
        auto raySample = mat.SampleBxDF(-wO, surf, rng);

        Spectrum refl;
        if(!Math::IsFinite(raySample.value.reflectance) || Math::IsNaN(raySample.pdf))
            refl = Spectrum(BIG_MAGENTA(), 0.0);
        else
            refl = DivideByPDF(raySample.value.reflectance, raySample.pdf);

        params.rayState.dOutputData[rayIndex] = refl;
    }
    else if(params.globalState.mode.e == Mode::AO)
    {
        Vector3 geoNormal;
        Vector3 normal;
        if constexpr(std::is_same_v<BasicSurface, Surface>)
            geoNormal = normal = surf.normal;
        else
        {
            normal = surf.shadingTBN.OrthoBasisZ();
            normal = Math::Normalize(tContext.ApplyN(normal));
            geoNormal = surf.geoNormal;
        }

        Vector2 xi = rng.NextFloat2D<0>();
        auto dirSample = Distribution::Common::SampleCosDirection(xi);
        Float NdL = Distribution::Common::DotN(dirSample.value);
        // From flat space (triangle laid out on XY plane) to directly world space
        Quaternion q = Quaternion::RotationBetweenZAxis(normal);
        Vector3 direction = q.ApplyRotation(dirSample.value);

        // Technically ao multiplier should be one after division by PDF
        // This is a simple shader, the division is explicitly specified
        // for verbosity.
        Vector3 aoMultiplier = Vector3(NdL * MathConstants::InvPi<Float>());
        aoMultiplier = Distribution::Common::DivideByPDF(aoMultiplier, dirSample.pdf);
        // Preset the ao multiplier, visibility check may override it after casting
        params.rayState.dOutputData[rayIndex] = Spectrum(aoMultiplier, 0);

        // New ray
        Ray rayOut = Ray(direction, surf.position).Nudge(geoNormal);
        Float tMax = params.globalState.tMaxAO;
        RayToGMem(params.rayState.dVisibilityRays, rayIndex, rayOut,
                  Vector2(0, tMax));
    }
    else params.rayState.dOutputData[rayIndex] = Spectrum(BIG_MAGENTA(), 0);
}

template<LightC Light, LightGroupC LG, TransformGroupC TG>
MR_HF_DEF
void SurfRDetail::LightWorkFunctionCommon(const Light&, RNGDispenser&,
                                          const RenderLightWorkParams<GlobalState,
                                                                      RayStateCommon, LG, TG>& params,
                                          RayIndex rayIndex)
{
    if constexpr (Light::IsPrimitiveBackedLight)
        params.rayState.dOutputData[rayIndex] = Spectrum(1, 1, 1, 0);
    else
        params.rayState.dOutputData[rayIndex] = Spectrum::Zero();
}
