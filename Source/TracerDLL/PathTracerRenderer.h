#pragma once

#include "Tracer/RendererC.h"
#include "Tracer/RayPartitioner.h"

#include "RequestedTypes.h" // IWYU pragma: keep
#include "PathTracerRendererShaders.h"
#include "SpectrumContext.h"

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

    MR_GF_DECL
    void operator()(uint32_t& i) const noexcept
    {
        i += constant;
    }
};

template<class SpectrumContextT>
class PathTracerRendererT final : public RendererT<PathTracerRendererT<SpectrumContextT>>
{
    using This              = PathTracerRendererT<SpectrumContextT>;
    using Base              = RendererT<This>;
    using FilmFilterPtr     = std::unique_ptr<TextureFilterI>;
    using CameraWorkPtr     = std::unique_ptr<RenderCameraWorkT<This>>;
    using Options           = PathTraceRDetail::Options;
    using SampleMode        = PathTraceRDetail::SampleMode;
    using LightSamplerType  = PathTraceRDetail::LightSamplerType;
    using RayState          = PathTraceRDetail::RayState;

    public:
    static std::string_view TypeName();
    static typename Base::AttribInfoList StaticAttributeInfo();
    //
    using UniformLightSampler   = DirectLightSamplerUniform<MetaLightList>;
    using AttribInfoList        = typename Base::AttribInfoList;
    using SpectrumContext       = typename SpectrumContextT;
    using SpectrumConverter     = typename SpectrumContext::Converter;
    static constexpr bool IsSpectral = std::is_same_v<SpectrumContext, SpectrumContextIdentity>;
    //
    using GlobalStateList       = TypePack<PathTraceRDetail::GlobalState<EmptyType>,
                                           PathTraceRDetail::GlobalState<UniformLightSampler>>;
    using RayStateList          = TypePack<PathTraceRDetail::RayState, PathTraceRDetail::RayState>;

    // Work Functions
    template<PrimitiveC P, MaterialC M, class S, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr auto WorkFunctions = Tuple
    {
        &PathTraceRDetail::WorkFunction<P, M, S, TContext, SpectrumConverter, PG, MG, TG>,
        &PathTraceRDetail::WorkFunctionNEE<UniformLightSampler, P, M, S, TContext, SpectrumConverter, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        &PathTraceRDetail::LightWorkFunction<L, SpectrumConverter, LG, TG>,
        &PathTraceRDetail::LightWorkFunctionWithNEE<UniformLightSampler, L, SpectrumConverter, LG, TG>
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};

    // Spectrum Converter Generator
    template<class GlobalState>
    MR_HF_DECL
    static SpectrumConverter GenSpectrumConverter(const GlobalState&, RayIndex rIndex);

    // On throughput mode, we do this burst, on latency mode
    // burst is implicit and is 1
    static constexpr uint32_t BurstSize = 32;

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    MetaLightList               metaLightArray;
    //
    FilmFilterPtr               filmFilter;
    RenderWorkHasher            workHasher;
    //
    Optional<CameraTransform>   curCamTransformOverride;
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const CameraWorkPtr*        curCamWork;
    std::vector<uint64_t>       tilePathCounts;
    std::vector<uint64_t>       tileSPPs;
    uint64_t                    totalDeadRayCount = 0;
    SampleMode                  anchorSampleMode;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rnGenerator;
    //
    DeviceMemory        rendererGlobalMem;
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<RayGMem>       dRays;
    Span<RayCone>       dRayCones;
    Span<uint32_t>      dShadowRayVisibilities;
    Span<RandomNumber>  dRandomNumBuffer;
    Span<Byte>          dSubCameraBuffer;
    Span<uint32_t>      dPathRNGDimensions;
    RayState            dRayState;
    // Work Hash related
    Span<CommonKey>     dWorkHashes;
    Span<CommonKey>     dWorkBatchIds;
    //
    bool                saveImage;
    //
    uint64_t            SPPLimit(uint32_t spp) const;
    uint32_t            FindMaxSamplePerIteration(uint32_t rayCount, PathTraceRDetail::SampleMode);

    Pair<Span<RayIndex>, uint32_t>
    ReloadPaths(Span<const RayIndex> dIndices,
                uint32_t sppLimit,
                const GPUQueue& processQueue);

    void                ResetAllPaths(const GPUQueue& queue);
    Span<RayIndex>      DoRenderPass(uint32_t sppLimit,
                                     const GPUQueue& queue);
    RendererOutput      DoThroughputSingleTileRender(const GPUDevice& device,
                                                     const GPUQueue& queue);
    RendererOutput      DoLatencyRender(uint32_t passCount,
                                        const GPUDevice& device,
                                        const GPUQueue& queue);

    public:
    // Constructors & Destructor
                         PathTracerRendererT(const RenderImagePtr&,
                                             TracerView,
                                             ThreadPool&,
                                             const GPUSystem&,
                                             const RenderWorkPack&);
                         PathTracerRendererT(const PathTracerRendererT&) = delete;
                         PathTracerRendererT(PathTracerRendererT&&) = delete;
    PathTracerRendererT& operator=(const PathTracerRendererT&) = delete;
    PathTracerRendererT& operator=(PathTracerRendererT&&) = delete;
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

extern template PathTracerRendererT<SpectrumContextIdentity>;
extern template PathTracerRendererT<SpectrumContextJakob2019>;

using PathTracerRenderer = PathTracerRendererT<SpectrumContextIdentity>;
using SpectralPathTracerRenderer = PathTracerRendererT<SpectrumContextJakob2019>;

static_assert(RendererC<PathTracerRenderer>, "\"PathTracerRenderer\" does not "
              "satisfy renderer concept.");

template<RendererC R, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
class PathTracerRenderWork : public RenderWork<R, PG, MG, TG>
{
    using Base = RenderWork<R, PG, MG, TG>;

    public:
    using Base::Base;

    uint32_t SampleRNCount(uint32_t workIndex) const override
    {
        constexpr uint32_t matSampleCount   = MG::template Material<>::SampleRNCount;
        constexpr uint32_t rrSampleCount    = 1;
        constexpr uint32_t lightSampleCount = R::UniformLightSampler::SampleLightRNCount;

        if(workIndex == 0)
            return (matSampleCount + rrSampleCount);
        else if(workIndex == 1)
            return lightSampleCount;
        return 0;
    }
};

template<RendererC R, LightGroupC LG, TransformGroupC TG>
using PathTracerRenderLightWork = RenderLightWork<R, LG, TG>;

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
using PathTracerRenderCamWork = RenderCameraWork<R, CG, TG>;

template<class SpectrumContext>
template<class GlobalState>
MR_HF_DEF
typename SpectrumContext::Converter
PathTracerRendererT<SpectrumContext>::GenSpectrumConverter(const GlobalState&, RayIndex)
{
    return SpectrumConverterIdentity();
}
