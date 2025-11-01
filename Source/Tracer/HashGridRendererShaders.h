#pragma once

#include "Random.h"
#include "HashGrid.h"
#include "SpectrumC.h"
#include "RendererC.h"
#include "RenderWork.h"

namespace HashGridRDetail
{
    enum class PathStatusEnum : uint8_t
    {
        // Path is dead (due to russian roulette or hitting a light source)
        DEAD = 0,
        // Hash is queried index converted to color
        // and written to the output buffer
        COLOR_WRITTEN = 1,
        //
        END
    };
    using PathStatus = Bitset<static_cast<size_t>(PathStatusEnum::END)>;

    struct GlobalState
    {
        HashGridView hashGrid;
        uint32_t     curDepth;
    };

    struct RayState
    {
        // Output related
        Span<Spectrum>          dPathColors;
        Span<ImageCoordinate>   dImageCoordinates;
        Span<Float>             dFilmFilterWeights;
        // Path state related
        Span<PathStatus>        dPathStatus;
        // Next set of rays
        Span<RayGMem>           dOutRays;
        Span<RayCone>           dOutRayCones;
    };

    template<PrimitiveGroupC PGType, MaterialGroupC MGType, TransformGroupC TGType>
    struct BounceWork
    {
        MRAY_WORK_FUNCTOR_DEFINE_TYPES(PGType, MGType, TGType,
                                       SpectrumContextIdentity, 1u);
        using Params = RenderWorkParams<GlobalState, RayState, PG, MG, TG>;

        MR_HF_DECL
        static void Call(const Primitive&, const Material&, const Surface&,
                         const RayConeSurface&, const TContext&,
                         const SpectrumConverterIdentity&, RNGDispenser&,
                         const Params& params, RayIndex rayIndex, uint32_t laneId);
    };

    template<LightGroupC LGType, TransformGroupC TGType>
    struct LightBounceWork
    {
        MRAY_LIGHT_WORK_FUNCTOR_DEFINE_TYPES(LGType, TGType,
                                             SpectrumContextIdentity, 1u);
        using Params = RenderLightWorkParams<GlobalState, RayState, LG, TG>;

        MR_HF_DECL
        static void Call(const Light&, RNGDispenser&,
                         const SpectrumConverterIdentity&,
                         const Params& params,
                         RayIndex rayIndex, uint32_t laneId);
    };
}

namespace HashGridRDetail
{

template<PrimitiveGroupC P, MaterialGroupC M, TransformGroupC T>
MR_HF_DEF
void BounceWork<P, M, T>::Call(const Primitive&, const Material& mat, const Surface& surf,
                               const RayConeSurface& surfRayCone, const TContext& tContext,
                               const SpectrumConv&, RNGDispenser& rng, const Params& params,
                               RayIndex rayIndex, uint32_t)
{
    PathStatus pathStatus = params.rayState.dPathStatus[rayIndex];

    // ================ //
    // Sample Material  //
    // ================ //
    auto [rayIn, tMM] = RayFromGMem(params.common.dRays, rayIndex);
    Vector3 wO = rayIn.dir;
    wO = Math::Normalize(tContext.InvApplyN(wO));
    auto raySample = mat.SampleBxDF(-wO, rng);
    Vector3 wI = Math::Normalize(tContext.ApplyN(raySample.wI.dir));
    Float specularity = mat.Specularity();

    // Refract the ray cone for russian roulette
    RayConeSurface rConeRefract = mat.RefractRayCone(surfRayCone, wO);
    RayCone rayConeOut = rConeRefract.ConeAfterScatter(wI, surf.geoNormal);

    // If material/surface is highly specular, just skip inserting the node
    // Global assumption here is that this region do not require
    // sophisticated approaches for LT estimation, it is just a portal to another
    // space in the scene.
    bool isSpecular = MaterialCommon::IsSpecular(specularity);
    if(!isSpecular)
    {
        const auto& hashGrid = params.globalState.hashGrid;
        BackupRNGState s = BackupRNG::GenerateState(rayIndex + 123456);
        BackupRNG rng2(s);

        Vector3 checkPos = surf.position;
        Vector3 checkNormal = surf.geoNormal;

        auto [spatialDataIndex, _] = hashGrid.TryInsertAtomic(checkPos, checkNormal);
        for(uint32_t ttt = 0; ttt < 4; ttt++)
        {
            Vector3 delta = Vector3((rng2.NextFloat() * 2 - 1) * 10,
                                    (rng2.NextFloat() * 2 - 1) * 10,
                                    (rng2.NextFloat() * 2 - 1) * 10);
            hashGrid.Search(checkPos + delta, surf.geoNormal);
        }

        if(!pathStatus[uint32_t(PathStatusEnum::COLOR_WRITTEN)])
        {
            pathStatus[uint32_t(PathStatusEnum::COLOR_WRITTEN)] = true;
            Spectrum color = Spectrum(Color::RandomColorRGB(spatialDataIndex), 0);
            params.rayState.dPathColors[rayIndex] = color;
        }
    }

    // Write back the new ray
    Vector3 nudgeNormal = surf.geoNormal;
    if(raySample.eval.isPassedThrough)
        nudgeNormal *= Float(-1);
    //
    Ray rayOut = Ray(wI, surf.position).Nudge(nudgeNormal);
    Vector2 tMMOut = Vector2(0, std::numeric_limits<Float>::max());
    RayToGMem(params.rayState.dOutRays, rayIndex, rayOut, tMMOut);
    params.rayState.dOutRayCones[rayIndex] = rayConeOut;
    params.rayState.dPathStatus[rayIndex] = pathStatus;
}

template<LightGroupC L, TransformGroupC T>
MR_HF_DEF
void LightBounceWork<L, T>::Call(const Light&, RNGDispenser&,
                                 const SpectrumConv&, const Params& params,
                                 RayIndex rayIndex, uint32_t)
{
    PathStatus pathStatus = params.rayState.dPathStatus[rayIndex];
    pathStatus[uint32_t(PathStatusEnum::DEAD)] = true;

    if constexpr(Light::IsPrimitiveBackedLight)
    {
        auto [ray, tMM] = RayFromGMem(params.common.dRays, rayIndex);
        RayCone rayCone = params.common.dRayCones[rayIndex].Advance(tMM[1]);
        Vector3 position = ray.AdvancedPos(tMM[1]);
        // What about normal?
        // We do not have surface information here
        // just fake normal as "wO"
        Vector3 wO = -ray.dir;

        const auto& hashGrid = params.globalState.hashGrid;
        auto [spatialDataIndex, _] = hashGrid.TryInsertAtomic(position, wO);

        if(!pathStatus[uint32_t(PathStatusEnum::COLOR_WRITTEN)])
        {
            pathStatus[uint32_t(PathStatusEnum::COLOR_WRITTEN)] = true;
            Spectrum color = Spectrum(Color::RandomColorRGB(spatialDataIndex), 0);
            params.rayState.dPathColors[rayIndex] = color;
        }
    }
    params.rayState.dPathStatus[rayIndex] = pathStatus;
}

}