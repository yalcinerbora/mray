#pragma once

#include "RendererC.h"
#include "SpectrumC.h"
#include "RayPartitioner.h"
#include "SurfaceRendererShaders.h"

class SurfaceRenderer final : public RendererT<SurfaceRenderer>
{
    using FilmFilterPtr = std::unique_ptr<TextureFilterI>;
    using CameraWorkPtr = std::unique_ptr<RenderCameraWorkT<SurfaceRenderer>>;
    using Options = SurfRDetail::Options;

    public:
    static std::string_view TypeName();
    static AttribInfoList StaticAttributeInfo();
    //
    using SpectrumContext = SpectrumContextIdentity;
    // Work States
    using GlobalStateList   = TypePack<SurfRDetail::GlobalState, SurfRDetail::GlobalState>;
    using RayStateList      = TypePack<SurfRDetail::RayStateCommon, SurfRDetail::RayStateAO>;
    using RayStateCommon    = TypePackElement<0, RayStateList>;
    using RayStateAO        = TypePackElement<1, RayStateList>;
    // Work Functions
    template<PrimitiveC P, MaterialC M, class S, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr auto WorkFunctions = Tuple
    {
        &SurfRDetail::WorkFunctionCommon<P, M, S, TContext, PG, MG, TG>,
        &SurfRDetail::WorkFunctionFurnaceOrAO<P, M, S, TContext, PG, MG, TG>
    };
    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = Tuple
    {
        &SurfRDetail::LightWorkFunctionCommon<L, LG, TG>
    };
    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = Tuple{};

    // Spectrum Converter Generator
    template<class RenderWorkParams>
    MR_HF_DECL
    static SpectrumConverterIdentity GenSpectrumConverter(const RenderWorkParams&, RayIndex);


    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    SurfRDetail::Mode::E        anchorMode;
    FilmFilterPtr               filmFilter;
    RenderWorkHasher            workHasher;
    //
    Optional<CameraTransform>   curCamTransformOverride;
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const CameraWorkPtr*        curCamWork;
    std::vector<uint64_t>       tilePathCounts;
    Float                       curTMaxAO = std::numeric_limits<Float>::max();
    //
    RayPartitioner      rayPartitioner;
    RNGeneratorPtr      rnGenerator;
    //
    DeviceMemory        rendererGlobalMem;
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<RayGMem>       dRays;
    Span<RayCone>       dRayCones;
    RayStateCommon      dRayStateCommon;
    RayStateAO          dRayStateAO;

    Span<uint32_t>      dIsVisibleBuffer;
    Span<RandomNumber>  dRandomNumBuffer;
    Span<Byte>          dSubCameraBuffer;

    // Work Hash related
    Span<CommonKey>     dWorkHashes;
    Span<CommonKey>     dWorkBatchIds;
    //
    bool                saveImage;

    uint32_t    FindMaxSamplePerIteration(uint32_t rayCount,
                                          SurfRDetail::Mode::E,
                                          bool doStochasticFilter);

    public:
    // Constructors & Destructor
                        SurfaceRenderer(const RenderImagePtr&,
                                        TracerView,
                                        ThreadPool&,
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
                                    uint32_t customLogicIndex0 = 0,
                                    uint32_t customLogicIndex1 = 0) override;
    RendererOutput      DoRender() override;
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

static_assert(RendererC<SurfaceRenderer>, "\"SurfaceRenderer\" does not "
              "satisfy renderer concept.");


template<RendererC R, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
class SurfaceRenderWork : public RenderWork<R, PG, MG, TG>
{
    using Base = RenderWork<R, PG, MG, TG>;

    public:
    using Base::Base;

    uint32_t SampleRNCount(uint32_t workIndex) const override
    {
        constexpr uint32_t matSampleCount = MG::template Material<>::SampleRNCount;
        if(workIndex == 1)
            return matSampleCount;
        return 0;
    }
};

template<RendererC R, LightGroupC LG, TransformGroupC TG>
using SurfaceRenderLightWork = RenderLightWork<R, LG, TG>;

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
using SurfaceRenderCamWork = RenderCameraWork<R, CG, TG>;

template <class RenderWorkParams>
MR_HF_DEF
SpectrumConverterIdentity
SurfaceRenderer::GenSpectrumConverter(const RenderWorkParams&, RayIndex)
{
    return SpectrumConverterIdentity();
}