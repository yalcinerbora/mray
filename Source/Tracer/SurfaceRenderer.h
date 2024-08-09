#pragma once

#include "RendererC.h"
#include "RenderWork.h"
#include "RayPartitioner.h"

#include "Core/TypeNameGenerators.h"

enum class RayType : uint8_t
{
    NEE_RAY,
    SPECULAR_PATH_RAY,
    PATH_RAY,
    CAMERA_RAY
};

//namespace PathRendererDetail
//{
//    struct RayState
//    {
//        Span<Spectrum>  dRadiance;
//        // Image coordinates of the path
//        Span<Vector2>   dImageCoordinates;
//        // Current path depth
//        Span<uint32_t>  dDepth;
//    };
//
//    struct RayPayload
//    {
//        // For NEE rays this is
//        // pre-calculated throughput
//        // as if the ray is hit to the light
//        //
//        // For path rays, this is normal throughput
//        Span<Spectrum>  dThroughput;
//
//        // NEE Related
//        // PDF of the material when ray goes
//        // toward the light direction (used for MIS)
//        Span<Float>     dMaterialPDF;
//        //
//        Span<RayType>   dRayType;
//        // Path is currently on this medium
//        Span<MediumKey> dCurrentMediumKeys;
//
//    };
//}

namespace SurfRDetail
{
    enum Mode
    {
        FURNACE,
        WORLD_NORMAL,
        WORLD_POSITION,
        AO,
        //
        END
    };

    struct GlobalState
    {
        // What are we rendering
        Mode mode;
    };

    struct RayState
    {
        // Can be position, furance radiance, normal
        // or a false color
        Span<Spectrum>  dOutputData;
        // Image coordinates of the path
        Span<Vector2>   dImageCoordinates;
    };
    // No payload (this is incident renderer so
    // everything is on ray state)
    using RayPayload = EmptyType;
}

class SurfaceRenderer final : public RendererT<SurfaceRenderer>
{
    public:
    static std::string_view TypeName();
    //
    using GlobalState   = SurfRDetail::GlobalState;
    using RayState      = SurfRDetail::RayState;
    using RayPayload    = SurfRDetail::RayPayload;
    using SpectrumConverterContext = SpectrumConverterContextIdentity;
    //
    struct Options
    {
        uint32_t            totalSPP   = 32;
        SurfRDetail::Mode   mode       = SurfRDetail::FURNACE;
    };

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    uint32_t    curTileIndex    = 0;
    Vector2ui   tileCount       = Vector2ui::Zero();
    //
    RenderImageParams           rIParams  = {};
    Optional<CameraTransform>   transOverride = {};
    CamSurfaceId                curCamSurfaceId = CamSurfaceId(0);
    const CameraGroupPtr*       currentCamera;
    //
    RayPartitioner              rayPartitioner;
    RNGeneratorPtr              rngGenerator;
    //
    DeviceMemory                rayStateMem;
    Span<MetaHit>               dHits;
    Span<HitKeyPack>            dHitKeys;
    Span<RayGMem>               dRays;
    Span<Byte>                  dSubCameraBuffer;

    public:
    // Constructors & Destructor
                        SurfaceRenderer(const RenderImagePtr&,
                                        const RenderWorkPack& wp,
                                        TracerView, const GPUSystem&);
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
                                    Optional<CameraTransform>,
                                    uint32_t customLogicIndex0 = 0,
                                    uint32_t customLogicIndex1 = 0) override;
    RendererOutput      DoRender() override;
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

inline
std::string_view SurfaceRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "Surface"sv;
    return RendererTypeName<Name>;
}

inline
size_t SurfaceRenderer::GPUMemoryUsage() const
{
    return (rayPartitioner.UsedGPUMemory() +
            rngGenerator->UsedGPUMemory() +
            rayStateMem.Size());
}

template<PrimitiveC Prim, MaterialC Material, class Surface,
         PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
MRAY_HYBRID
void SurfaceRendWorkFunction(const Prim& prim, const Material& mat, const Surface& surf,
                             const RenderWorkParams<SurfaceRenderer, PG, MG, TG>& params)
{
    //
}

template<LightC Light,
         LightGroupC LG, TransformGroupC TG>
MRAY_HYBRID
void SurfaceRendLightWorkFunction(const Light& light,
                                  const RenderLightWorkParams<SurfaceRenderer, LG, TG>& params)
{
    //
}

template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
MRAY_HYBRID
void SurfaceRendCameraWorkFunction(const Camera&,
                                   const RenderCameraWorkParams<SurfaceRenderer, CG, TG>&)
{
    // Empty, no notion of ray hitting camera
}

template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
using SurfaceRenderWork = RenderWork<SurfaceRenderer, PG, MG, TG>;

template<LightGroupC LG, TransformGroupC TG>
using SurfaceRenderLightWork = RenderLightWork<SurfaceRenderer, LG, TG>;

template<CameraGroupC CG, TransformGroupC TG>
using SurfaceRenderCamWork = RenderCameraWork<SurfaceRenderer, CG, TG>;
