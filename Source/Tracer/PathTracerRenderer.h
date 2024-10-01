#pragma once

#include "RendererC.h"
#include "RayPartitioner.h"
#include "PathTracerRendererShaders.h"

class PathTracerRenderer final : public RendererT<PathTracerRenderer>
{
    using FilmFilterPtr = std::unique_ptr<TextureFilterI>;
    using CameraWorkPtr = std::unique_ptr<RenderCameraWorkT<PathTracerRenderer>>;

    public:
    static std::string_view TypeName();
    //
    using GlobalState   = PathTraceRDetail::GlobalState;
    using RayState      = PathTraceRDetail::RayState;
    using RayPayload    = PathTraceRDetail::RayPayload;
    using SpectrumConverterContext = SpectrumConverterContextIdentity;
    using Options       = PathTraceRDetail::Options;
    // Work Functions
    template<PrimitiveC P, MaterialC M, class S, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr Tuple WorkFunctions = Tuple
    {
        PathTraceRDetail::WorkFunction<P, M, S, TContext, PG, MG, TG>,
        PathTraceRDetail::WorkFunctionWithNEE<P, M, S, TContext, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        PathTraceRDetail::LightWorkFunction<L, LG, TG>
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};
    static constexpr auto RayStateInitFunc = PathTraceRDetail::InitRayState;

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    FilmFilterPtr               filmFilter;
    RenderWorkHasher            workHasher;
    //
    Optional<CameraTransform>   curCamTransformOverride;
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const CameraWorkPtr*        curCamWork;
    uint64_t                    globalPixelIndex = 0;
    Float                       curTMaxAO = std::numeric_limits<Float>::max();
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

    uint32_t    FindMaxSamplePerIteration(uint32_t rayCount, PathTraceRDetail::SampleMode);

    public:
    // Constructors & Destructor
                            PathTracerRenderer(const RenderImagePtr&,
                                               TracerView,
                                               BS::thread_pool&,
                                               const GPUSystem&,
                                               const RenderWorkPack&);
                            PathTracerRenderer(const PathTracerRenderer&) = delete;
                            PathTracerRenderer(PathTracerRenderer&&) = delete;
    PathTracerRenderer&     operator=(const PathTracerRenderer&) = delete;
    PathTracerRenderer&     operator=(PathTracerRenderer&&) = delete;
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

static_assert(RendererC<PathTracerRenderer>);

template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
using PathTracerRenderWork = RenderWork<PathTracerRenderer, PG, MG, TG>;

template<LightGroupC LG, TransformGroupC TG>
using PathTracerRenderLightWork = RenderLightWork<PathTracerRenderer, LG, TG>;

template<CameraGroupC CG, TransformGroupC TG>
using PathTracerRenderCamWork = RenderCameraWork<PathTracerRenderer, CG, TG>;