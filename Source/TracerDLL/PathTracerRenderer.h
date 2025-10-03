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
class ConstAddFunctor_U16
{
    private:
    uint16_t constant;

    public:
    ConstAddFunctor_U16(uint16_t c) : constant(c) {}

    MR_GF_DECL
    void operator()(uint16_t& i) const noexcept
    {
        i += constant;
    }
};

class SetFunctor_U16
{
    private:
    uint16_t constant;

    public:
    SetFunctor_U16(uint16_t c) : constant(c) {}

    MR_GF_DECL
    void operator()(uint16_t& i) const noexcept
    {
        i = constant;
    }
};

template<SpectrumContextC SpectrumContextT>
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
    using SpecContextPtr        = std::unique_ptr<SpectrumContext>;
    static constexpr bool IsSpectral = !std::is_same_v<SpectrumContext, SpectrumContextIdentity>;
    //
    using GlobalStateList       = TypePack<PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>,
                                           PathTraceRDetail::GlobalState<UniformLightSampler, SpectrumConverter>>;
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
    template<class RenderWorkParams>
    MR_HF_DECL
    static SpectrumConverter GenSpectrumConverter(const RenderWorkParams&, RayIndex rIndex);

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
    Span<uint16_t>      dPathRNGDimensions;
    RayState            dRayState;
    // Work Hash related
    Span<CommonKey>     dWorkHashes;
    Span<CommonKey>     dWorkBatchIds;
    //
    bool                saveImage;
    //
    uint64_t            SPPLimit(uint32_t spp) const;
    uint32_t            FindMaxSamplePerIteration(uint32_t rayCount, PathTraceRDetail::SampleMode);
    //
    SpecContextPtr      spectrumContext;
    bool                isSpectral = IsSpectral;
    Span<Spectrum>      dSpectrumWavePDFs;

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

using PathTracerRendererRGB = PathTracerRendererT<SpectrumContextIdentity>;
using PathTracerRendererSpectral = PathTracerRendererT<SpectrumContextJakob2019>;

static_assert(RendererC<PathTracerRendererRGB>, "\"PathTracerRendererRGB\" does not "
              "satisfy renderer concept.");
static_assert(RendererC<PathTracerRendererSpectral>, "\"PathTracerRendererSpectral\" does not "
              "satisfy renderer concept.");

template<RendererC R, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
class PathTracerRenderWork : public RenderWork<R, PG, MG, TG>
{
    using Base = RenderWork<R, PG, MG, TG>;

    public:
    using Base::Base;

    RNRequestList SampleRNList(uint32_t workIndex) const override
    {
        static constexpr auto matSampleList   = MG::template Material<>::SampleRNList;
        static constexpr auto rrSampleList    = GenRNRequestList<1>();
        static constexpr auto lightSampleList = R::UniformLightSampler::SampleLightRNList;
        //
        if(workIndex == 0)  return matSampleList.Append(rrSampleList);
        if(workIndex == 1)  return lightSampleList;
                            return RNRequestList();
    }
};

template<RendererC R, LightGroupC LG, TransformGroupC TG>
using PathTracerRenderLightWork = RenderLightWork<R, LG, TG>;

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
using PathTracerRenderCamWork = RenderCameraWork<R, CG, TG>;

template<SpectrumContextC SpectrumContext>
template<class RenderWorkParams>
MR_HF_DEF
typename SpectrumContext::Converter
PathTracerRendererT<SpectrumContext>::GenSpectrumConverter(const RenderWorkParams& params,
                                                           RayIndex rIndex)
{
    if constexpr(!std::is_same_v<SpectrumContext, SpectrumContextIdentity>)
    {
        return SpectrumConverter(params.rayState.dPathWavelengths[rIndex],
                                 params.globalState.specContextData);
    }
    else return SpectrumConverterIdentity();
}
