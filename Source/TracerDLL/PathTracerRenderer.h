#pragma once

#include "Tracer/PathTracerRendererBase.h"
#include "Tracer/MediaTracker.h"

#include "RequestedTypes.h" // IWYU pragma: keep
#include "PathTracerRendererShaders.h"
#include "SpectrumContext.h"

template<SpectrumContextC SpectrumContextT>
class PathTracerRendererT final : public PathTracerRendererBase
{
    using This              = PathTracerRendererT<SpectrumContextT>;
    using Base              = PathTracerRendererBase;
    using Options           = PathTraceRDetail::Options;
    using SampleMode        = PathTraceRDetail::SampleMode;
    using RayState          = PathTraceRDetail::RayState;

    public:
    //
    using MediaTrackerPtr       = std::unique_ptr<MediaTracker>;
    using UniformLightSampler   = DirectLightSamplerUniform<MetaLightList>;
    using AttribInfoList        = typename RendererBase::AttribInfoList;
    using SpectrumContext       = SpectrumContextT;
    using SpectrumConverter     = typename SpectrumContext::Converter;
    static constexpr bool IsSpectral = !std::is_same_v<SpectrumContext, SpectrumContextIdentity>;
    //
    using GlobalStateList = TypePack
    <
        PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>,
        PathTraceRDetail::GlobalState<UniformLightSampler, SpectrumConverter>,
        PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>
    >;
    using RayStateList = TypePack
    <
        PathTraceRDetail::RayState,
        PathTraceRDetail::RayState,
        PathTraceRDetail::RayState
    >;

    // Work Functions
    template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkFunctions = TypePack
    <
        PathTraceRDetail::WorkFunction<PG, MG, TG, SpectrumContext>,
        PathTraceRDetail::WorkFunctionNEE<PG, MG, TG, SpectrumContext, UniformLightSampler>
    >;
    template<LightGroupC LG, TransformGroupC TG>
    using LightWorkFunctions = TypePack
    <
        PathTraceRDetail::LightWorkFunction<LG, TG, SpectrumContext>,
        PathTraceRDetail::LightWorkFunctionWithNEE<LG, TG, SpectrumContext, UniformLightSampler>
    >;
    template<CameraGroupC CG, TransformGroupC TG>
    using CamWorkFunctions = TypePack<>;

    template<MediumGroupC MG, TransformGroupC TG>
    using MediumWorkFunctions = TypePack
    <
        PathTraceRDetail::MediumWorkFunction<MG, TG, SpectrumContext>,
        PathTraceRDetail::MediumWorkFunctionWithNEE<MG, TG, SpectrumContext, UniformLightSampler>,
        PathTraceRDetail::MediumWorkFunctionTransmittance<MG, TG, SpectrumContext>
    >;

    static std::string_view TypeName();
    static AttribInfoList   StaticAttributeInfo();

    // Spectrum Converter Generator
    template<class RenderWorkParams>
    MR_HF_DECL
    static SpectrumConverter GenSpectrumConverter(const RenderWorkParams&, RayIndex rIndex);

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    MetaLightList       metaLightArray;
    // Memory
    DeviceMemory        rendererGlobalMem;
    // Work Hash related
    RenderSurfaceWorkHasher surfaceWorkHasher;
    Span<CommonKey>         dSurfaceWorkHashes;
    Span<CommonKey>         dSurfaceWorkBatchIds;
    RenderMediumWorkHasher  mediumWorkHasher;
    Span<CommonKey>         dMediumWorkHashes;
    Span<CommonKey>         dMediumWorkBatchIds;
    // Ray-cast Related
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<uint32_t>      dShadowRayVisibilities;
    //
    Span<RayGMem>       dShadowRays;
    Span<RayCone>       dShadowRayCones;
    Span<Spectrum>      dShadowRayRadiance;
    //
    Span<Float>         dPrevMatPDF;
    //
    bool                saveImage  = false;
    // Media Related
    MediaTrackerPtr         mediaTracker;
    Span<RayMediaListPack>  dRayMediaListPacks;

    // Helpers
    uint32_t            FindMaxSamplePerIteration(uint32_t rayCount, PathTraceRDetail::SampleMode);
    Span<RayIndex>      DoRenderPassPure(Span<RayIndex>, Span<CommonKey>,
                                         const GPUQueue&);
    Span<RayIndex>      DoRenderPassNEE(Span<RayIndex>, Span<CommonKey>,
                                        const GPUQueue&);
    Span<RayIndex>      DoRenderPassWithMediaPure(Span<RayIndex>, Span<CommonKey>,
                                                  const GPUQueue&);
    Span<RayIndex>      DoRenderPassWithMediaNEE(Span<RayIndex>, Span<CommonKey>,
                                                 const GPUQueue&);
    Span<RayIndex>      DoRenderPass(uint32_t sppLimit, const GPUQueue&);
    void                RecursiveShadowRayCast(Bitspan<uint32_t> isVisibleBitSpan,
                                               // I-O
                                               Span<BackupRNGState> dBackupRNGStates,
                                               Span<const RayIndex> dRayIndices,
                                               const GPUQueue& queue);
    // Implementations
    RendererOutput      DoThroughputSingleTileRender(const GPUDevice&, const GPUQueue&) override;
    RendererOutput      DoLatencyRender(uint32_t passCount, const GPUDevice&,
                                        const GPUQueue&) override;

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
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

extern template class PathTracerRendererT<SpectrumContextIdentity>;
extern template class PathTracerRendererT<SpectrumContextJakob2019>;

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
             if(workIndex == 0) return matSampleList.Append(rrSampleList);
        else if(workIndex == 1) return lightSampleList;
        else                    return RNRequestList();
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
