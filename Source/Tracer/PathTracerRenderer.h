#pragma once

#include "RendererC.h"
#include "RendererCommon.h"
#include "RayPartitioner.h"
#include "PathTracerRendererShaders.h"

#include "Core/Timer.h"

// Due to NVCC error
// "An extended __host__ __device__
// lambda cannot be defined inside a generic lambda expression."
//
// We make this a functor
// TODO: This is somewhat related to the RNG (at least currently)
// maybe change this later.
class ConstAddFunctor
{
    private:
    uint32_t constant;

    public:
    ConstAddFunctor(uint32_t c) : constant(c) {}

    MRAY_HYBRID MRAY_GPU_INLINE
    uint32_t operator()(uint32_t i) const
    {
        return i + constant;
    }
};

#include "Core/Flag.h"

template<class MetaLightArrayType>
class PathTracerRenderer final : public RendererT<PathTracerRenderer<MetaLightArrayType>>
{

    using Base              = RendererT<PathTracerRenderer<MetaLightArrayType>>;
    using FilmFilterPtr     = std::unique_ptr<TextureFilterI>;
    using CameraWorkPtr     = std::unique_ptr<RenderCameraWorkT<PathTracerRenderer>>;
    using MetaLightArray    = MetaLightArrayType;
    using Options           = PathTraceRDetail::Options;
    using SampleMode        = PathTraceRDetail::SampleMode;
    using LightSamplerType  = PathTraceRDetail::LightSamplerType;
    using RayState          = PathTraceRDetail::RayState;
    //
    using UniformLightSampler   = DirectLightSamplerViewUniform<typename MetaLightArray::View>;

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
    uint64_t                    totalDeadRayCount = 0;
    SampleMode                  anchorSampleMode;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rnGenerator;
    //
    DeviceMemory        redererGlobalMem;
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<RayGMem>       dRays;
    Span<RayDiff>       dRayDifferentials;
    Span<uint32_t>      dShadowRayVisibilities;
    Span<RandomNumber>  dRandomNumBuffer;
    Span<Byte>          dSubCameraBuffer;
    Span<uint32_t>      dPathRNGDimensions;
    RayState            dRayState;
    // Work Hash related
    Span<CommonKey>     dWorkHashes;
    Span<CommonKey>     dWorkBatchIds;

    uint32_t            FindMaxSamplePerIteration(uint32_t rayCount, PathTraceRDetail::SampleMode);
    Span<RayIndex>      ReloadPaths(Span<const RayIndex> dIndices,
                                    const GPUQueue& processQueue);
    void                ResetAllPaths(const GPUQueue& queue);

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

template<RendererC R, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
class PathTracerRenderWork : public RenderWork<R, PG, MG, TG>
{
    using Base = RenderWork<R, PG, MG, TG>;

    public:
    using Base::Base;

    uint32_t SampleRNCount(uint32_t workIndex) const override
    {
        constexpr uint32_t matSampleCount = MG::template Material<>::SampleRNCount;
        constexpr uint32_t rrSampleCount = 1;
        // TODO: Get this programatically?
        constexpr uint32_t lightSampleCount = 2;

        if(workIndex == 0)
            return (matSampleCount + rrSampleCount);
        else if(workIndex == 1)
            return (matSampleCount + rrSampleCount + lightSampleCount);

        return 0;
    }
};

template<RendererC R, LightGroupC LG, TransformGroupC TG>
using PathTracerRenderLightWork = RenderLightWork<R, LG, TG>;

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
using PathTracerRenderCamWork = RenderCameraWork<R, CG, TG>;

#include "PathTracerRenderer.hpp"
