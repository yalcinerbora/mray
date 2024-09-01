#pragma once

#include "RendererC.h"
#include "RenderWork.h"
#include "RayPartitioner.h"
#include "TextureFilter.h"

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
        Span<Spectrum>          dOutputData;
        Span<ImageCoordinate>   dImageCoordinates;
    };
    // No payload (this is incident renderer so
    // everything is on ray state)
    using RayPayload = EmptyType;

    template<PrimitiveC Prim, MaterialC Material, class Surface,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    MRAY_HYBRID
    void WorkFunction(const Prim& prim, const Material& mat, const Surface& surf,
                                 const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params);

    template<LightC Light, LightGroupC LG, TransformGroupC TG>
    MRAY_HYBRID
    void LightWorkFunction(const Light& light,
                                      const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params);

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
    // Work Functions
    template<PrimitiveC P, MaterialC M, class S,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr Tuple WorkFunctions = Tuple
    {
        SurfRDetail::WorkFunction<P, M, S, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        SurfRDetail::LightWorkFunction<L, LG, TG>
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};

    static constexpr auto RayStateInitFunc = SurfRDetail::InitRayState;

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
    FilmFilterPtr               filmFilter;
    RenderWorkHasher            workHasher;
    //
    Optional<CameraTransform>   transOverride = {};
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const CameraWorkPtr*        curCamWork;
    uint64_t                    globalPixelIndex = 0;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rnGenerator;
    //
    DeviceMemory                redererGlobalMem;
    Span<MetaHit>               dHits;
    Span<HitKeyPack>            dHitKeys;
    Span<RayGMem>               dRays;
    Span<RayDiff>               dRayDifferentials;
    Span<RandomNumber>          dCamGenRandomNums;
    Span<Byte>                  dSubCameraBuffer;
    RayState                    dRayState;
    // Work Hash related
    Span<uint32_t>              dWorkHashes;
    Span<CommonKey>             dWorkBatchIds;

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
            rnGenerator->UsedGPUMemory() +
            redererGlobalMem.Size());
}

template<PrimitiveC Prim, MaterialC Material, class Surface,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::WorkFunction(const Prim& prim, const Material& mat, const Surface& surf,
                               const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params)
{
    //
}

template<LightC Light,
         LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID
void SurfRDetail::LightWorkFunction(const Light& light,
                                    const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params)
{
    //
}

MRAY_HYBRID MRAY_CGPU_INLINE
void SurfRDetail::InitRayState(const RayPayload&,
                               const RayState& dStates,
                               const RaySample& raySample,
                               uint32_t writeIndex)
{
    dStates.dImageCoordinates[writeIndex] = raySample.value.imgCoords;
}

template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
using SurfaceRenderWork = RenderWork<SurfaceRenderer, PG, MG, TG>;

template<LightGroupC LG, TransformGroupC TG>
using SurfaceRenderLightWork = RenderLightWork<SurfaceRenderer, LG, TG>;

template<CameraGroupC CG, TransformGroupC TG>
using SurfaceRenderCamWork = RenderCameraWork<SurfaceRenderer, CG, TG>;
