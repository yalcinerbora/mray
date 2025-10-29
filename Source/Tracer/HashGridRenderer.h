#pragma once

#include "RendererC.h"
#include "SpectrumC.h"
#include "Random.h"
#include "RenderWork.h"
#include "HashGrid.h"
#include "RayPartitioner.h"

#include "Core/BitFunctions.h"
#include "Core/ColorFunctions.h"

#include "HashGridRendererShaders.h"

class HashGridRenderer final : public RendererBase
{
    public:
    static std::string_view TypeName();
    static AttribInfoList StaticAttributeInfo();

    using RayState          = HashGridRDetail::RayState;
    using GlobalState       = HashGridRDetail::GlobalState;
    using GlobalStateList   = TypePack<GlobalState>;
    using RayStateList      = TypePack<RayState>;
    using SpectrumContext   = SpectrumContextIdentity;

    template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkFunctions = TypePack
    <
        HashGridRDetail::BounceWork<PG, MG, TG>
    >;

    template<LightGroupC LG, TransformGroupC TG>
    using LightWorkFunctions = TypePack
    <
        HashGridRDetail::LightBounceWork<LG, TG>
    >;

    template<CameraGroupC CG, TransformGroupC TG>
    using CamWorkFunctions = TypePack<>;

    template<class RenderWorkParams>
    MR_HF_DECL
    static SpectrumConverterIdentity GenSpectrumConverter(const RenderWorkParams&, RayIndex);

    struct Options
    {
        uint32_t cacheEntryLimit   = 3'000'000;
        uint32_t cachePosBits      = 12;
        uint32_t cacheNormalBits   = 2;
        uint32_t cacheLevelCount   = 4;
        Float    cacheConeAperture = Float(0.6);
        //
        uint32_t pathTraceDepth    = 3;
    };

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    // State
    uint32_t    curPosBits    = 0;
    uint32_t    curNormalBits = 0;
    //
    bool        saveImage;
    // Camera stuff
    Optional<CameraTransform>   curCamTransformOverride;
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const RenderCameraWorkI*    curCamWork;
    // Renderer Systems and Memory
    HashGrid            hashGrid;
    RayPartitioner      rayPartitioner;
    RNGeneratorPtr      rnGenerator;
    RenderWorkHasher    workHasher;
    //
    DeviceMemory        rendererGlobalMem;
    Span<MetaHit>       dHits;
    Span<HitKeyPack>    dHitKeys;
    Span<RayGMem>       dRays;
    Span<RayCone>       dRayCones;
    Span<RandomNumber>  dRandomNumBuffer;
    Span<Byte>          dSubCameraBuffer;
    Span<uint16_t>      dPathRNGDimensions;
    RayState            dRayState;
    // Work Hash related
    Span<CommonKey>     dWorkHashes;
    Span<CommonKey>     dWorkBatchIds;
    // For hash grid, cam position buffer
    Span<Vector3>       dCamPosBuffer;

    uint32_t    FindMaxSamplePerIteration(uint32_t maxRayCount);
    void        PathTraceAndQuery();
    bool        CopyAliveRays(uint32_t rayCount, uint32_t maxWorkCount,
                              const GPUQueue& processQueue);

    public:
    // Constructors & Destructor
    using This = HashGridRenderer;
            HashGridRenderer(const RenderImagePtr&,
                             TracerView,
                             ThreadPool&,
                             const GPUSystem&,
                             const RenderWorkPack&);
            HashGridRenderer(const This&) = delete;
            HashGridRenderer(This&&) = delete;
    This&   operator=(const This&) = delete;
    This&   operator=(This&&) = delete;

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

static_assert(RendererC<HashGridRenderer>, "\"HashGridRenderer\" does not "
              "satisfy renderer concept.");

template<RendererC R, PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
class HashGridRenderWork : public RenderWork<R, PG, MG, TG>
{
    using Base = RenderWork<R, PG, MG, TG>;

    public:
    using Base::Base;

    RNRequestList SampleRNList(uint32_t workIndex) const override
    {
        static constexpr auto matSampleList = MG::template Material<>::SampleRNList;
        //
        if(workIndex == 0) return matSampleList;
        else               return RNRequestList();
    }
};

template<RendererC R, LightGroupC LG, TransformGroupC TG>
using HashGridRenderLightWork = RenderLightWork<R, LG, TG>;

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
using HashGridRenderCamWork = RenderCameraWork<R, CG, TG>;

template <class RenderWorkParams>
MR_HF_DEF
SpectrumConverterIdentity
HashGridRenderer::GenSpectrumConverter(const RenderWorkParams&, RayIndex)
{
    return SpectrumConverterIdentity();
}