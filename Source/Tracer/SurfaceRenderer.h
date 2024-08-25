#pragma once

#include "RendererC.h"
#include "RenderWork.h"
#include "RayPartitioner.h"

#include "Core/TypeNameGenerators.h"

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
    enum Mode
    {
        FURNACE,
        WORLD_NORMAL,
        WORLD_POSITION,
        AO,
        //
        END
    };

    struct GlobalState
    {
        // What are we rendering
        Mode mode;
    };

    struct RayState
    {
        // Can be position, furance radiance, normal
        // or a false color
        Span<Spectrum>  dOutputData;
        // Image coordinates of the path
        Span<Vector2>   dImageCoordinates;
    };
    // No payload (this is incident renderer so
    // everything is on ray state)
    using RayPayload = EmptyType;

    template<PrimitiveC Prim, MaterialC Material, class Surface,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void SurfaceRendWorkFunction(const Prim& prim, const Material& mat, const Surface& surf,
                                 const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void SurfaceRendLightWorkFunction(const Light& light,
                                      const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params);

    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    MRAY_HYBRID
    void SurfaceRendCameraWorkFunction(const Camera&,
                                       const RenderCameraWorkParams<SurfaceRenderer, CG, TG>&);

    MRAY_HYBRID
    void SurfaceRendInitRayPayload(const RayPayload&,
                                   uint32_t writeIndex, const RaySample&);
}

class SurfaceRenderer final : public RendererT<SurfaceRenderer>
{
    public:
    static std::string_view TypeName();
    //
    using GlobalState   = SurfRDetail::GlobalState;
    using RayState      = SurfRDetail::RayState;
    using RayPayload    = SurfRDetail::RayPayload;
    using SpectrumConverterContext = SpectrumConverterContextIdentity;
    // Work Functions
    template<PrimitiveC P, MaterialC M, class S,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr Tuple WorkFunctions = Tuple
    {
        SurfRDetail::SurfaceRendWorkFunction<P, M, S, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        SurfRDetail::SurfaceRendLightWorkFunction<L, LG, TG>
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};

    static constexpr auto RayStateInitFunc = SurfRDetail::SurfaceRendInitRayPayload;

    //
    struct Options
    {
        uint32_t            totalSPP   = 32;
        SurfRDetail::Mode   mode       = SurfRDetail::FURNACE;
    };

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    uint32_t    curTileIndex    = 0;
    Vector2ui   tileCount       = Vector2ui::Zero();
    //
    RenderImageParams           rIParams  = {};
    Optional<CameraTransform>   transOverride = {};
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const RenderCameraWorkPtr*  curCamWork;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rngGenerator;
    //
    DeviceMemory                rayStateMem;
    Span<MetaHit>               dHits;
    Span<HitKeyPack>            dHitKeys;
    Span<RayGMem>               dRays;
    Span<RayDiff>               dRayDifferentials;
    Span<Byte>                  dSubCameraBuffer;
    RayState                    dRayState;

    public:
    // Constructors & Destructor
                        SurfaceRenderer(const RenderImagePtr&,
                                        const RenderWorkPack& wp,
                                        TracerView, const GPUSystem&);
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
                                    Optional<CameraTransform>,
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
            rngGenerator->UsedGPUMemory() +
            rayStateMem.Size());
}

template<PrimitiveC Prim, MaterialC Material, class Surface,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::SurfaceRendWorkFunction(const Prim& prim, const Material& mat, const Surface& surf,
                                          const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params)
{
    //
}

template<LightC Light,
         LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::SurfaceRendLightWorkFunction(const Light& light,
                                               const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params)
{
    //
}

template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::SurfaceRendCameraWorkFunction(const Camera&,
                                                const RenderCameraWorkParams<SurfaceRenderer, CG, TG>&)
{
    // Empty, no notion of ray hitting camera
}

MRAY_HYBRID MRAY_CGPU_INLINE
void SurfRDetail::SurfaceRendInitRayPayload(const RayPayload&, uint32_t,
                                            const RaySample&)
{
    // TODO: ....
}

template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
using SurfaceRenderWork = RenderWork<SurfaceRenderer, PG, MG, TG>;

template<LightGroupC LG, TransformGroupC TG>
using SurfaceRenderLightWork = RenderLightWork<SurfaceRenderer, LG, TG>;

template<CameraGroupC CG, TransformGroupC TG>
using SurfaceRenderCamWork = RenderCameraWork<SurfaceRenderer, CG, TG>;
