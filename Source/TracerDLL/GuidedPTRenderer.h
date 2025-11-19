#pragma once

#include "GuidedPTRendererShaders.h"

class GuidedPTRenderer final : public PathTracerRendererBase
{
    using This          = GuidedPTRenderer;
    using Base          = PathTracerRendererBase;
    using Options       = GuidedPTRDetail::Options;
    using DisplayMode   = GuidedPTRDetail::DisplayMode;
    using RayState      = GuidedPTRDetail::RayState;
    using GlobalState   = GuidedPTRDetail::GlobalState;
    //
    using MCState       = GuidedPTRDetail::MCState;
    using MCLock        = GuidedPTRDetail::MCLock;
    using MCCount       = GuidedPTRDetail::MCCount;
    using MCIrradiance  = GuidedPTRDetail::MCIrradiance;

    public:
    //
    using ImageSectionOpt       = Optional<RenderImageSection>;
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

    template<MediumGroupC MG, TransformGroupC TG>
    using MediumWorkFunctions = TypePack<>;

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
    HashGrid            hashGrid;
    // Memory
    DeviceMemory        rendererGlobalMem;
    // Work Hash related
    RenderSurfaceWorkHasher workHasher;
    Span<CommonKey>         dWorkHashes;
    Span<CommonKey>         dWorkBatchIds;
    // Ray-cast Related
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<uint32_t>      dShadowRayVisibilities;
    // NEE Related
    Span<RayGMem>       dShadowRays;
    Span<RayCone>       dShadowRayCones;
    Span<Spectrum>      dShadowRayRadiance;
    // Technique Related (Per-path)
    Span<Float>         dShadowPrevPathOutRadiance;
    Span<Float>         dPrevPathReflectanceOrOutRadiance;
    Span<Float>         dPrevPDF;
    Span<Vector2>       dPrevNormals;
    Span<Float>         dScoreSums;
    Span<uint32_t>      dLiftedMCIndices;
    // Technique Related (Hash Grid)
    Span<MCState>       dMCStates;
    Span<MCLock>        dMCLocks;
    Span<MCCount>       dMCCounts;
    Span<MCIrradiance>  dMCIrradiances;
    // For hash grid, cam position buffer
    Span<Vector3>       dCamPosBuffer;
    //
    bool                saveImage  = false;

    // Helpers
    RayState            PackRayState() const;
    GlobalState         PackGlobalState() const;
    ImageSectionOpt     DisplayHashGrid(Span<const RayIndex> dDeadRayIndices,
                                        const GPUQueue& processQueue,
                                        const GPUQueue& transferQueue);
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
        static constexpr auto RNList = []()
        {
            constexpr auto matSampleList = MG::template Material<>::SampleRNList;
            constexpr auto lobeSampleList = GuidedPTRDetail::GaussianLobeMixture::SampleRNList;
            constexpr auto lightSampleList = R::UniformLightSampler::SampleLightRNList;
            // Material or Lobe MIS sample
            auto list = matSampleList.Append(lobeSampleList);
            // Russian Rouletted Sample
            list = list.Append(GenRNRequestList<1>());
            // MIS Sample (Material or Lobe Selection)
            list = list.Append(GenRNRequestList<1>());
            // NEE Sample
            list = list.Append(lightSampleList);
            return list;
        }();
        //
        if(workIndex != 0) return RNRequestList();
        //

        return RNList;
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