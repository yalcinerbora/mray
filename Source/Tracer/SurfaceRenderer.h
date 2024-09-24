#pragma once

#include "RendererC.h"
#include "RenderWork.h"
#include "RayPartitioner.h"
#include "TextureFilter.h"

#include "Core/TypeNameGenerators.h"
#include "Core/ColorFunctions.h"

class SurfaceRenderer;

enum class RayType : uint8_t
{
    NEE_RAY,
    SPECULAR_PATH_RAY,
    PATH_RAY,
    CAMERA_RAY
};

namespace SurfRDetail
{

    struct Mode
    {
        public:
        enum E
        {
            AO,
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
        Float       tMaxAO              = std::numeric_limits<Float>::max();
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

    template<PrimitiveC Prim, MaterialC Material,
             class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void WorkFunctionCommon(const Prim&, const Material&, const Surface&,
                            const TContext&, RNGDispenser&,
                            const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
                            RayIndex rayIndex);

    template<PrimitiveC Prim, MaterialC Material,
             class Surface, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void WorkFunctionAO(const Prim&, const Material&, const Surface&,
                        const TContext&, RNGDispenser&,
                        const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
                        RayIndex rayIndex);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void LightWorkFunctionCommon(const Light&, RNGDispenser&,
                                 const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params,
                                 RayIndex rayIndex);
    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void LightWorkFunctionAO(const Light&, RNGDispenser&,
                             const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params,
                             RayIndex rayIndex);

    MRAY_HYBRID
    void InitRayState(const RayPayload&, const RayState&,
                      const RaySample&, uint32_t writeIndex);
}

class SurfaceRenderer final : public RendererT<SurfaceRenderer>
{
    using FilmFilterPtr = std::unique_ptr<TextureFilterI>;
    using CameraWorkPtr = std::unique_ptr<RenderCameraWorkT<SurfaceRenderer>>;

    public:
    static std::string_view TypeName();
    //
    using GlobalState   = SurfRDetail::GlobalState;
    using RayState      = SurfRDetail::RayState;
    using RayPayload    = SurfRDetail::RayPayload;
    using SpectrumConverterContext = SpectrumConverterContextIdentity;
    using Options       = SurfRDetail::Options;
    // Work Functions
    template<PrimitiveC P, MaterialC M, class S, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr Tuple WorkFunctions = Tuple
    {
        SurfRDetail::WorkFunctionCommon<P, M, S, TContext, PG, MG, TG>,
        SurfRDetail::WorkFunctionAO<P, M, S, TContext, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        SurfRDetail::LightWorkFunctionCommon<L, LG, TG>,
        SurfRDetail::LightWorkFunctionAO<L, LG, TG>,
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};
    static constexpr auto RayStateInitFunc = SurfRDetail::InitRayState;

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    SurfRDetail::Mode::E        anchorMode;
    FilmFilterPtr               filmFilter;
    RenderWorkHasher            workHasher;
    //
    Optional<CameraTransform>   curCamTransformOverride;
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const CameraWorkPtr*        curCamWork;
    uint64_t                    globalPixelIndex = 0;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rnGenerator;
    //
    DeviceMemory                    redererGlobalMem;
    Span<MetaHit>                   dHits;
    Span<HitKeyPack>                dHitKeys;
    std::array<Span<RayGMem>, 2>    dRays;
    std::array<Span<RayDiff>, 2>    dRayDifferentials;

    Span<uint32_t>              dIsVisibleBuffer;
    Span<RandomNumber>          dRandomNumBuffer;
    Span<Byte>                  dSubCameraBuffer;
    RayState                    dRayState;
    // Work Hash related
    Span<uint32_t>              dWorkHashes;
    Span<CommonKey>             dWorkBatchIds;

    uint32_t    FindMaxSamplePerIteration(uint32_t rayCount, SurfRDetail::Mode::E);

    public:
    // Constructors & Destructor
                        SurfaceRenderer(const RenderImagePtr&,
                                        TracerView,
                                        BS::thread_pool&,
                                        const GPUSystem&,
                                        const RenderWorkPack&);
                        SurfaceRenderer(const SurfaceRenderer&) = delete;
                        SurfaceRenderer(SurfaceRenderer&&) = delete;
    SurfaceRenderer&    operator=(const SurfaceRenderer&) = delete;
    SurfaceRenderer&    operator=(SurfaceRenderer&&) = delete;
    //
    AttribInfoList      AttributeInfo() const override;
    RendererOptionPack  CurrentAttributes() const override;
    void                PushAttribute(uint32_t attributeIndex,
                                      TransientData data,
                                      const GPUQueue& q) override;
    //
    RenderBufferInfo    StartRender(const RenderImageParams&,
                                    CamSurfaceId camSurfId,
                                    uint32_t customLogicIndex0 = 0,
                                    uint32_t customLogicIndex1 = 0) override;
    RendererOutput      DoRender() override;
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

inline
std::string_view SurfaceRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "Surface"sv;
    return RendererTypeName<Name>;
}

inline
size_t SurfaceRenderer::GPUMemoryUsage() const
{
    return (rayPartitioner.UsedGPUMemory() +
            rnGenerator->UsedGPUMemory() +
            redererGlobalMem.Size());
}

class PrimGroupTriangle;
class PrimGroupSkinnedTriangle;

template<PrimitiveC Prim, MaterialC Material,
         class Surface, class TContext,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID
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
            if constexpr(std::is_same_v<PrimGroupTriangle, PG>  ||
                         std::is_same_v<PrimGroupSkinnedTriangle, PG>)
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
MRAY_HYBRID
void SurfRDetail::WorkFunctionAO(const Prim&, const Material&, const Surface& surf,
                                 const TContext& tContext, RNGDispenser& rng,
                                 const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params,
                                 RayIndex rayIndex)
{
    assert(params.globalState.mode.e == Mode::AO);

    Vector3 normal;
    if constexpr(std::is_same_v<BasicSurface, Surface>)
        normal = surf.normal;
    else
    {
        normal = surf.shadingTBN.ApplyInvRotation(Vector3::ZAxis());
        normal = tContext.ApplyN(normal).Normalize();
    }
    Vector3 position = surf.position;

    Vector2 xi = rng.NextFloat2D<0>();
    auto dirSample = Distribution::Common::SampleCosDirection(xi);
    // From flat space (triangle laid out on XY plane) to directly world space
    Quaternion q = Quaternion::RotationBetweenZAxis(normal);
    Vector3 direction = q.ApplyRotation(dirSample.value);
    Float nDotL = direction.Dot(normal);

    // Technically ao multiplier should be one after division by PDF
    // This is a simple shader, the division is explicitly specified
    // for verbosity.
    Vector3 aoMultiplier = Vector3(nDotL * MathConstants::InvPi<Float>());
    aoMultiplier = Distribution::Common::DivideByPDF(aoMultiplier, dirSample.pdf);
    // Preset the ao multiplier, visibility check may override it after casting
    params.rayState.dOutputData[rayIndex] = Spectrum(aoMultiplier, 0);

    // New ray
    Ray rayOut = Ray(direction, position);
    rayOut.NudgeSelf(normal);
    Float tMax = params.globalState.tMaxAO;
    RayToGMem(params.out.dRays, rayIndex, rayOut, Vector2(0, tMax));
}

template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::LightWorkFunctionCommon(const Light& l, RNGDispenser&,
                                          const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params,
                                          RayIndex rayIndex)
{
    if(l.IsPrimitiveBackedLight())
        params.rayState.dOutputData[rayIndex] = Spectrum(1, 1, 1, 0);
    else
        params.rayState.dOutputData[rayIndex] = Spectrum::Zero();
}


template<LightC Light, LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::LightWorkFunctionAO(const Light&, RNGDispenser&,
                                      const RenderLightWorkParams<SurfaceRenderer, LG, TG>&,
                                      RayIndex)
{
    //if(l.IsPrimitiveBackedLight())
    //    params.rayState.dOutputData[rayIndex] = Spectrum(1, 1, 1, 0);
    //else
    //    params.rayState.dOutputData[rayIndex] = Spectrum::Zero();
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

template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
using SurfaceRenderWork = RenderWork<SurfaceRenderer, PG, MG, TG>;

template<LightGroupC LG, TransformGroupC TG>
using SurfaceRenderLightWork = RenderLightWork<SurfaceRenderer, LG, TG>;

template<CameraGroupC CG, TransformGroupC TG>
using SurfaceRenderCamWork = RenderCameraWork<SurfaceRenderer, CG, TG>;
