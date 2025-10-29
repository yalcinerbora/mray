#pragma once

#include "Tracer/RendererC.h"
#include "Tracer/RayPartitioner.h"
#include "SpectrumContext.h"

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

struct PathTracerRendererBase : public RendererBase
{
    public:
    struct ReloadPathOutput
    {
        Span<RayIndex> dIndices;
        uint32_t aliveRayCount;
    };
    using RenderImageSectionOpt = Optional<RenderImageSection>;

    private:
    using FilmFilterPtr = std::unique_ptr<TextureFilterI>;
    // On throughput mode, we do this burst, on latency mode
    // burst is implicit and is 1
    static constexpr uint32_t BurstSize = 32u;
    //
    FilmFilterPtr               filmFilter;
    RenderWorkHasher            workHasher;
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
    uint32_t                    totalSPP;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rnGenerator;

    // =============================== //
    //  Camera Ray Generation Related  //
    // =============================== //
    // If spectral mode is enabled, these must be set.
    const SpectrumContextI*     spectrumContext;
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
    RendererAnalyticData  CalculateAnalyticDataThroughput(size_t deadRayCount, const Timer& timer);

    RendererAnalyticData  CalculateAnalyticDataLatency(uint32_t passPathCount, const Timer& timer);
    ReloadPathOutput      ReloadPaths(Span<const RayIndex> dIndices,
                                      uint32_t sppLimit, const GPUQueue& processQueue);
    void                  ResetAllPaths(const GPUQueue& queue);
    Span<RayIndex>        DoRenderPass(uint32_t sppLimit, const GPUQueue& queue);
    RenderImageSectionOpt AddRadianceToRenderBufferThroughput(Span<const RayIndex> dDeadRayIndices,
                                                              const GPUQueue& processQueue,
                                                              const GPUQueue& transferQueue);
    void                  AddRadianceToRenderBufferLatency(Span<const RayIndex> dDeadRayIndices,
                                                           const GPUQueue& processQueue);
    void                  InitializeForRender(CamSurfaceId camSurfId,
                                              uint32_t maxRayCount,
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
