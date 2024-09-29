#pragma once

#include <array>
#include <string_view>
#include "Core/ColorFunctions.h"

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
            "AmbientOcculusion",
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
    void WorkFunctionCommon(const Prim&, const Material&, const Surface&,
                            const TContext&, RNGDispenser&,
                            const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
                            RayIndex rayIndex);

    template<PrimitiveC Prim, MaterialC Material, class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void WorkFunctionFurnaceOrAO(const Prim&, const Material&, const Surface&,
                                 const TContext&, RNGDispenser&,
                                 const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
                                 RayIndex rayIndex);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void LightWorkFunctionCommon(const Light&, RNGDispenser&,
                                 const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params,
                                 RayIndex rayIndex);

    MRAY_HYBRID
    void InitRayState(const RayPayload&, const RayState&,
                      const RaySample&, uint32_t writeIndex);
}

template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void SurfRDetail::WorkFunctionCommon(const Prim&, const Material&, const Surface& surf,
                                     const TContext& tContext, RNGDispenser&,
                                     const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
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
                Vector3 normal = surf.shadingTBN.ApplyInvRotation(Vector3::ZAxis());
                normal = tContext.ApplyN(normal).Normalize();
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
            color = Color::RandomColorRGB(static_cast<uint32_t>(mKey));
            break;
        }
        case PRIM_ID:
        {
            PrimitiveKey pKey = params.in.dKeys[rayIndex].primKey;
            color = Color::RandomColorRGB(static_cast<uint32_t>(pKey));
            break;
        }
        case ACCEL_ID:
        {
            AcceleratorKey aKey = params.in.dKeys[rayIndex].accelKey;
            color = Color::RandomColorRGB(static_cast<uint32_t>(aKey));
            break;
        }
        case TRANSFORM_ID:
        {
            TransformKey tKey = params.in.dKeys[rayIndex].transKey;
            color = Color::RandomColorRGB(static_cast<uint32_t>(tKey));
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
        default: color = BIG_MAGENTA; break;
    }
    params.rayState.dOutputData[rayIndex] = Spectrum(color, Float(0));
}

template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
void SurfRDetail::WorkFunctionFurnaceOrAO(const Prim&, const Material& mat, const Surface& surf,
                                          const TContext& tContext, RNGDispenser& rng,
                                          const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
                                          RayIndex rayIndex)
{
    assert(params.globalState.mode.e == Mode::AO ||
           params.globalState.mode.e == Mode::FURNACE);

    if(params.globalState.mode.e == Mode::FURNACE)
    {
        using Distribution::Common::DivideByPDF;
        auto [rayIn, tMM] = RayFromGMem(params.in.dRays, rayIndex);
        Vector3 wI = rayIn.Dir();
        wI = tContext.InvApplyN(wI).Normalize();
        auto raySample = mat.SampleBxDF(-wI, surf, rng);

        Spectrum refl;
        if(raySample.value.reflectance.HasNaN() || Math::IsNan(raySample.pdf))
            refl = Spectrum(BIG_MAGENTA, 0.0);
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
            normal = surf.shadingTBN.ApplyInvRotation(Vector3::ZAxis());
            geoNormal = surf.geoNormal;
        }
        normal = tContext.ApplyN(normal).Normalize();

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
        Ray rayOut = Ray(direction, surf.position);
        rayOut.NudgeSelf(geoNormal);
        Float tMax = params.globalState.tMaxAO;
        RayToGMem(params.out.dRays, rayIndex, rayOut, Vector2(0, tMax));
    }
    else params.rayState.dOutputData[rayIndex] = Spectrum(BIG_MAGENTA, 0);
}

template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID MRAY_GPU_INLINE
void SurfRDetail::LightWorkFunctionCommon(const Light& l, RNGDispenser&,
                                          const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params,
                                          RayIndex rayIndex)
{
    if(l.IsPrimitiveBackedLight())
        params.rayState.dOutputData[rayIndex] = Spectrum(1, 1, 1, 0);
    else
        params.rayState.dOutputData[rayIndex] = Spectrum::Zero();
}

MRAY_HYBRID MRAY_CGPU_INLINE
void SurfRDetail::InitRayState(const RayPayload&,
                               const RayState& dStates,
                               const RaySample& raySample,
                               uint32_t writeIndex)
{
    dStates.dImageCoordinates[writeIndex] = raySample.value.imgCoords;
    dStates.dFilmFilterWeights[writeIndex] = raySample.pdf;
}
