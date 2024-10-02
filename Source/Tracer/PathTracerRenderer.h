#pragma once

#include "RendererC.h"
#include "RayPartitioner.h"
#include "PathTracerRendererShaders.h"

template<class MetaLightArrayType>
class PathTracerRenderer final : public RendererT<PathTracerRenderer<MetaLightArrayType>>
{
    using Base              = RendererT<PathTracerRenderer<MetaLightArrayType>>;
    using FilmFilterPtr     = std::unique_ptr<TextureFilterI>;
    using CameraWorkPtr     = std::unique_ptr<RenderCameraWorkT<PathTracerRenderer>>;
    using MetaLightArray    = MetaLightArrayType;
    using Options           = PathTraceRDetail::Options;
    //
    using UniformLightSampler   = DirectLightSamplerViewUniform<typename MetaLightArray::View>;
    //using IrradianceLightSampler    = DirectLightSamplerViewUniform<typename MetaLightArray::View>;

    public:
    static std::string_view TypeName();
    //
    using AttribInfoList = typename Base::AttribInfoList;
    using SpectrumConverterContext = SpectrumConverterContextIdentity;
    using GlobalStateList   = Tuple<PathTraceRDetail::GlobalState<EmptyType>,
                                    PathTraceRDetail::GlobalState<UniformLightSampler>>;
    using RayStateList      = Tuple<PathTraceRDetail::RayState, PathTraceRDetail::RayState>;

    // Work Functions
    template<PrimitiveC P, MaterialC M, class S, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr Tuple WorkFunctions = Tuple
    {
        PathTraceRDetail::WorkFunction<P, M, S, TContext, PG, MG, TG>,
        PathTraceRDetail::WorkFunctionWithNEE<UniformLightSampler, P, M, S, TContext, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        PathTraceRDetail::LightWorkFunction<L, LG, TG>,
        PathTraceRDetail::LightWorkFunctionWithNEE<UniformLightSampler, L, LG, TG>
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    MetaLightArray              metaLightArray;
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
    PathTraceRDetail::RayState  dRayState;
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

#include "PathTracerRenderer.hpp"
