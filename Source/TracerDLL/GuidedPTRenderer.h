#pragma once

#include "GuidedPTRendererShaders.h"

class GuidedPTRenderer final : public PathTracerRendererBase
{
    using This              = GuidedPTRenderer;
    using Base              = PathTracerRendererBase;
    using Options           = GuidedPTRDetail::Options;
    using DisplayMode       = GuidedPTRDetail::DisplayMode;
    using RayState          = GuidedPTRDetail::RayState;

    public:
    //
    using UniformLightSampler   = DirectLightSamplerUniform<MetaLightList>;
    using AttribInfoList        = typename RendererBase::AttribInfoList;
    using SpectrumContext       = SpectrumContextJakob2019;
    using SpectrumConverter     = typename SpectrumContext::Converter;
    static constexpr bool IsSpectral = !std::is_same_v<SpectrumContext, SpectrumContextIdentity>;

    using GlobalStateList       = TypePack<GuidedPTRDetail::GlobalState>;
    using RayStateList          = TypePack<GuidedPTRDetail::RayState>;

    // Work Functions
    template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkFunctions = TypePack
    <
        GuidedPTRDetail::WorkFunction<PG, MG, TG>
    >;
    template<LightGroupC LG, TransformGroupC TG>
    using LightWorkFunctions = TypePack
    <
        GuidedPTRDetail::LightWorkFunction<LG, TG>
    >;
    template<CameraGroupC CG, TransformGroupC TG>
    using CamWorkFunctions = TypePack<>;

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
    RenderWorkHasher    workHasher;
    Span<CommonKey>     dWorkHashes;
    Span<CommonKey>     dWorkBatchIds;
    // Ray-cast Related
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<uint32_t>      dShadowRayVisibilities;
    // NEE Related
    Span<RayGMem>       dShadowRays;
    Span<RayCone>       dShadowRayCones;
    Span<Spectrum>      dShadowRayRadiance;
    Span<Float>         dPrevMatPDF;
    Span<Float>         dPrevAvgReflectance;
    // Technique Related
    Span<Float>         dPDFChains;
    Span<uint32_t>      dLiftedMarkovChainIndex;

    //
    bool                saveImage  = false;

    // Helpers
    //void                CopyAliveRays(Span<const RayIndex> dAliveRayIndices, const GPUQueue&);
    uint32_t            FindMaxSamplePerIteration(uint32_t rayCount);
    Span<RayIndex>      DoRenderPass(uint32_t sppLimit, const GPUQueue&);
    // Implementations
    RendererOutput      DoThroughputSingleTileRender(const GPUDevice&, const GPUQueue&) override;
    RendererOutput      DoLatencyRender(uint32_t passCount, const GPUDevice&,
                                        const GPUQueue&) override;

    public:
    // Constructors & Destructor
                         GuidedPTRenderer(const RenderImagePtr&,
                                          TracerView,
                                          ThreadPool&,
                                          const GPUSystem&,
                                          const RenderWorkPack&);
                         GuidedPTRenderer(const This&) = delete;
                         GuidedPTRenderer(This&&) = delete;
    This&                operator=(const This&) = delete;
    This&                operator=(This&&) = delete;
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

static_assert(RendererC<GuidedPTRenderer>, "\"GuidedPTRenderer\" does not "
              "satisfy renderer concept.");

template<RendererC R, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
class GuidedPTRenderWork : public RenderWork<R, PG, MG, TG>
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
using GuidedPTRenderLightWork = RenderLightWork<R, LG, TG>;

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
using GuidedPTRenderCamWork = RenderCameraWork<R, CG, TG>;

template<class RenderWorkParams>
MR_HF_DEF
typename SpectrumContextJakob2019::Converter
GuidedPTRenderer::GenSpectrumConverter(const RenderWorkParams& params,
                                       RayIndex rIndex)
{
    return SpectrumConverter(params.rayState.dPathWavelengths[rIndex],
                             params.globalState.specContextData);
}

// TODO: We may add not spectral version later
using GuidedPTRendererSpectral = GuidedPTRenderer;