#pragma once

#include "Tracer/RendererC.h"
#include "Tracer/RayPartitioner.h"
#include "SpectrumC.h"

#include "Core/NamedEnum.h"

class Timer;

enum class RayType : uint8_t
{
    SHADOW_RAY,
    SPECULAR_RAY,
    PATH_RAY,
    CAMERA_RAY
};

enum class PathStatusEnum : uint8_t
{
    // Path is dead (due to russian roulette or hitting a light source)
    DEAD = 0,
    // Invalid rays are slightly different, sometimes due to exactly
    // meeting the spp requirement, renderer may not launch rays,
    // it will mark these as invalid in the pool
    INVALID = 1,
    // TODO: These are not used yet, but here for future use.
    // Path did scatter because of the medium. It should not go
    // Material scattering
    MEDIUM_SCATTERED = 2,
    // TODO: Maybe incorporate ray type?
    //
    END
};
using PathStatus = Bitset<static_cast<size_t>(PathStatusEnum::END)>;

enum class RenderModeEnum
{
    THROUGHPUT,
    LATENCY,
    //
    END
};
inline constexpr std::array RenderModeNames =
{
    "Throughput",
    "Latency"
};
using RenderMode = NamedEnum<RenderModeEnum, RenderModeNames>;

struct alignas(4) PathDataPack
{
    uint8_t     depth;
    PathStatus  status;
    RayType     type;
};

struct SetPathStateFunctor
{
    PathStatusEnum e;
    bool firstInit;

    MRAY_HOST inline
    SetPathStateFunctor(PathStatusEnum eIn, bool firstInit = false)
        : e(eIn)
        , firstInit(firstInit)
    {}

    MR_HF_DECL
    void operator()(PathDataPack& s) const
    {
        if(firstInit) s.status.Reset();

        if(e == PathStatusEnum::INVALID)
            s.status.Set(uint32_t(PathStatusEnum::DEAD), false);
        else if(e == PathStatusEnum::DEAD)
            s.status.Set(uint32_t(PathStatusEnum::INVALID), false);
        //
        s.status.Set(uint32_t(e));
    }
};

class IsDeadAliveInvalidFunctor
{
    Span<const PathDataPack> dPathDataPack;

    public:
    IsDeadAliveInvalidFunctor(Span<const PathDataPack> dPathDataPackIn)
        : dPathDataPack(dPathDataPackIn)
    {}

    MR_HF_DECL
    uint32_t operator()(RayIndex index) const noexcept
    {
        const PathStatus state = dPathDataPack[index].status;

        bool isDead = state[uint32_t(PathStatusEnum::DEAD)];
        bool isInvalid = state[uint32_t(PathStatusEnum::INVALID)];
        assert(!isDead || !isInvalid);

        if(isDead)          return 0;
        else if(isInvalid)  return 1;
        else                return 2;
    }
};

class IsAliveFunctor
{
    Span<const PathDataPack> dPathDataPack;
    bool checkInvalidAsDead;

    public:
    IsAliveFunctor(Span<const PathDataPack> dPathDataPackIn,
                   bool checkInvalidAsDeadIn = false)
        : dPathDataPack(dPathDataPackIn)
        , checkInvalidAsDead(checkInvalidAsDeadIn)
    {}

    MR_HF_DECL
    bool operator()(RayIndex index) const
    {
        const PathStatus state = dPathDataPack[index].status;
        bool result = state[uint32_t(PathStatusEnum::DEAD)];
        if(checkInvalidAsDead)
            result = result || state[uint32_t(PathStatusEnum::INVALID)];
        return !result;
    }
};


struct PathTracerRendererBase : public RendererBase
{
    public:
    struct ReloadPathOutput
    {
        Span<RayIndex> dIndices;
        uint32_t aliveRayCount;
    };
    struct InitOutput
    {
        uint32_t maxRayCount;
        uint32_t totalWorkCount;
    };

    using RenderImageSectionOpt = Optional<RenderImageSection>;

    protected:
    using FilmFilterPtr = std::unique_ptr<TextureFilterI>;
    using SpectrumContextPtr = std::unique_ptr<SpectrumContextI>;
    // On throughput mode, we do this burst, on latency mode
    // burst is implicit and is 1
    static constexpr uint32_t BurstSize = 32u;
    //
    FilmFilterPtr               filmFilter;
    //
    Optional<CameraTransform>   curCamTransformOverride;
    CameraSurfaceParams         curCamSurfaceParams;
    TransformKey                curCamTransformKey;
    CameraKey                   curCamKey;
    const RenderCameraWorkI*    curCamWork;
    std::vector<uint64_t>       tilePathCounts;
    std::vector<uint64_t>       tileSPPs;
    uint64_t                    totalDeadRayCount = 0;
    RenderMode                  renderMode;
    uint32_t                    burstSize;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rnGenerator;

    // =============================== //
    //  Camera Ray Generation Related  //
    // =============================== //
    // If spectral mode is enabled, these must be set.
    SpectrumContextPtr          spectrumContext;
    Span<SpectrumWaves>         dPathWavelengths;
    Span<Spectrum>              dSpectrumWavePDFs;
    // ================================ //
    //  Common Data Allocated By Parent //
    // ================================ //
    Span<PathDataPack>    dPathDataPack;
    Span<RandomNumber>    dRandomNumBuffer;
    Span<RayCone>         dRayCones;
    Span<ImageCoordinate> dImageCoordinates;
    Span<Float>           dFilmFilterWeights;
    Span<Spectrum>        dThroughputs;
    Span<Spectrum>        dPathRadiance;
    Span<RayGMem>         dRays;
    Span<uint16_t>        dPathRNGDimensions;
    Span<Byte>            dSubCameraBuffer;

    // Helpers
    uint64_t              TotalSampleLimit(uint32_t spp) const;
    RendererAnalyticData  CalculateAnalyticDataThroughput(size_t deadRayCount, uint32_t totalSPP,
                                                          const Timer& timer);

    RendererAnalyticData  CalculateAnalyticDataLatency(uint32_t passPathCount, uint32_t totalSPP,
                                                       const Timer& timer) const;
    ReloadPathOutput      ReloadPaths(Span<const RayIndex> dIndices,
                                      uint32_t sppLimit, const GPUQueue& processQueue);
    void                  ResetAllPaths(const GPUQueue& queue);
    Span<RayIndex>        DoRenderPass(uint32_t sppLimit, const GPUQueue& queue);
    RenderImageSectionOpt AddRadianceToRenderBufferThroughput(Span<const RayIndex> dDeadRayIndices,
                                                              const GPUQueue& processQueue,
                                                              const GPUQueue& transferQueue);
    void                  AddRadianceToRenderBufferLatency(Span<const RayIndex> dDeadRayIndices,
                                                           const GPUQueue& processQueue) const;
    InitOutput            InitializeForRender(CamSurfaceId camSurfId,
                                              uint32_t sppLimit,
                                              bool retainCameraTransform,
                                              const RenderImageParams& renderImgParams);

    // Implemented by the path tracers
    virtual RendererOutput DoThroughputSingleTileRender(const GPUDevice& device,
                                                        const GPUQueue& queue) = 0;
    virtual RendererOutput DoLatencyRender(uint32_t passCount,
                                           const GPUDevice& device,
                                           const GPUQueue& queue) = 0;

    public:
    // Constructors & Destructor
                            PathTracerRendererBase(const RenderImagePtr&, TracerView,
                                                   ThreadPool&, const GPUSystem&,
                                                   const RenderWorkPack&,
                                                   std::string_view rendererName);
                             PathTracerRendererBase(const PathTracerRendererBase&) = delete;
                             PathTracerRendererBase(PathTracerRendererBase&&) = delete;
    PathTracerRendererBase&  operator=(const PathTracerRendererBase&) = delete;
    PathTracerRendererBase&  operator=(PathTracerRendererBase&&) = delete;

    // Impl.
    RendererOutput      DoRender() override;

};
