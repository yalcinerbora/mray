#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "TracerConstants.h"
#include "TracerAttribInfo.h"

#include "Common/RenderImageStructs.h"

// Only TransientData is exposed to the user
namespace TransientPoolDetail
{
    class TransientData;
}
using TransientData = TransientPoolDetail::TransientData;

// std::numeric_limits<IdType>::max() returns **ZERO** !!!!
// This should be reported I think?
// enum class Id : uint32_t {} effectively create a new integral type
// (This is how std byte is defined for example)
// and maximum of Byte should be 255 (or whatever the maximum of byte size of that platform)
// We do convert it to underlying type to call max here.
template<class IdType>
// TODO: This is c++23 (Can we implement this ourselves?)
//requires std::is_scoped_enum_v<IdType>
inline constexpr IdType TracerIdInvalid =  IdType(std::numeric_limits<std::underlying_type_t<IdType>>::max());

static_assert(std::numeric_limits<std::byte>::max() == std::byte(0), "Really?");

class TimelineSemaphore;
class ThreadPool;

using GPUThreadInitFunction = void(*)();

struct RenderImageParams
{
    Vector2ui               resolution;
    Vector2ui               regionMin;
    Vector2ui               regionMax;
};

struct TracerParameters
{
    // Random Seed value, many samplers etc.
    // will start via this seed if applicable
    uint64_t            seed = 0;
    // Accelerator mode, software/hardware etc.
    AcceleratorType     accelMode = AcceleratorType::HARDWARE;
    // Item pool size, amount of "items" (paths/rays/etc) processed
    // in parallel
    uint32_t            parallelizationHint = 1 << 21; // 2^21 ~= 2M
    // Current sampler logic,
    SamplerType         samplerType = SamplerType::INDEPENDENT;
    // Texture Related
    uint32_t            clampedTexRes = std::numeric_limits<uint32_t>::max();
    bool                genMips = false;
    FilterType          mipGenFilter = FilterType{ FilterType::GAUSSIAN, 2.0f };
    MRayColorSpaceEnum  globalTextureColorSpace = MRayColorSpaceEnum::MR_ACES_CG;
    // Film Related
    FilterType          filmFilter = FilterType{FilterType::GAUSSIAN, 1.0f};
    // Spectral Rendering Related
    WavelengthSampleMode wavelengthSampleMode = WavelengthSampleMode::HYPERBOLIC_PBRT;
};

// Generic Texture Input Parameters
struct MRayTextureParameters
{
    MRayPixelTypeRT             pixelType;
    MRayColorSpaceEnum          colorSpace      = MRayColorSpaceEnum::MR_DEFAULT;
    Float                       gamma           = Float(1);
    bool                        ignoreResClamp  = false;
    AttributeIsColor            isColor         = AttributeIsColor::IS_COLOR;
    MRayTextureEdgeResolveEnum  edgeResolve     = MRayTextureEdgeResolveEnum::MR_WRAP;
    MRayTextureInterpEnum       interpolation   = MRayTextureInterpEnum::MR_LINEAR;
    MRayTextureReadMode         readMode        = MRayTextureReadMode::MR_PASSTHROUGH;
};

// For surface commit analytic information
struct SurfaceCommitResult
{
    AABB3 aabb;
    size_t instanceCount;
    size_t acceleratorCount;
};

using MaterialIdList    = std::vector<MaterialId>;
using TransformIdList   = std::vector<TransformId>;
using MediumIdList      = std::vector<MediumId>;
using LightIdList       = std::vector<LightId>;
using CameraIdList      = std::vector<CameraId>;

namespace TracerConstants
{
    // Implicit Mediums that are always present on a tracer system
    static constexpr MediumGroupId VacuumMediumGroupId  = MediumGroupId(0);
    static constexpr MediumId VacuumMediumId            = MediumId(0);

    static constexpr LightGroupId NullLightGroupId      = LightGroupId(0);
    static constexpr LightId NullLightId                = LightId(0);

    static constexpr TransGroupId IdentityTransGroupId  = TransGroupId(0);
    static constexpr TransformId IdentityTransformId    = TransformId(0);

    static constexpr PrimGroupId EmptyPrimGroupId       = PrimGroupId{0};
    static constexpr PrimBatchId EmptyPrimBatchId       = PrimBatchId{0};

    static constexpr TextureId InvalidTexture           = TextureId{0};
    static constexpr MediumPair VacuumMediumPair        = Pair(VacuumMediumId, VacuumMediumId);

    static const auto NoAlphaMapList = OptionalAlphaMapList(StaticVecSize(MaxPrimBatchPerSurface),
                                                            std::nullopt);
    static const auto CullFaceTrueList = CullBackfaceFlagList(StaticVecSize(MaxPrimBatchPerSurface),
                                                              true);
};

struct SurfaceParams
{
    SurfacePrimList         primBatches;
    SurfaceMatList          materials;
    TransformId             transformId;
    OptionalAlphaMapList    alphaMaps;
    CullBackfaceFlagList    cullFaceFlags;
};

struct LightSurfaceParams
{
    LightId     lightId;
    TransformId transformId = TracerConstants::IdentityTransformId;
    MediumId    mediumId    = TracerConstants::VacuumMediumId;
};

struct CameraSurfaceParams
{
    CameraId    cameraId;
    TransformId transformId = TracerConstants::IdentityTransformId;
    MediumId    mediumId    = TracerConstants::VacuumMediumId;
};

class [[nodiscard]] TracerI
{
    public:
    virtual     ~TracerI() = default;

    //================================//
    //            Generic             //
    //================================//
    virtual TypeNameList        PrimitiveGroups() const = 0;
    virtual TypeNameList        MaterialGroups() const = 0;
    virtual TypeNameList        TransformGroups() const = 0;
    virtual TypeNameList        CameraGroups() const = 0;
    virtual TypeNameList        MediumGroups() const = 0;
    virtual TypeNameList        LightGroups() const = 0;
    virtual TypeNameList        Renderers() const = 0;

    virtual PrimAttributeInfoList       AttributeInfo(PrimGroupId) const = 0;
    virtual CamAttributeInfoList        AttributeInfo(CameraGroupId) const = 0;
    virtual MediumAttributeInfoList     AttributeInfo(MediumGroupId) const = 0;
    virtual MatAttributeInfoList        AttributeInfo(MatGroupId) const = 0;
    virtual TransAttributeInfoList      AttributeInfo(TransGroupId) const = 0;
    virtual LightAttributeInfoList      AttributeInfo(LightGroupId) const = 0;
    virtual RendererAttributeInfoList   AttributeInfo(RendererId) const = 0;

    virtual PrimAttributeInfoList       AttributeInfoPrim(std::string_view) const = 0;
    virtual CamAttributeInfoList        AttributeInfoCam(std::string_view) const = 0;
    virtual MediumAttributeInfoList     AttributeInfoMedium(std::string_view) const = 0;
    virtual MatAttributeInfoList        AttributeInfoMat(std::string_view) const = 0;
    virtual TransAttributeInfoList      AttributeInfoTrans(std::string_view) const = 0;
    virtual LightAttributeInfoList      AttributeInfoLight(std::string_view) const = 0;
    virtual RendererAttributeInfoList   AttributeInfoRenderer(std::string_view) const = 0;

    virtual std::string                 TypeName(PrimGroupId) const = 0;
    virtual std::string                 TypeName(CameraGroupId) const = 0;
    virtual std::string                 TypeName(MediumGroupId) const = 0;
    virtual std::string                 TypeName(MatGroupId) const = 0;
    virtual std::string                 TypeName(TransGroupId) const = 0;
    virtual std::string                 TypeName(LightGroupId) const = 0;
    virtual std::string                 TypeName(RendererId) const = 0;

    //================================//
    //           Primitive            //
    //================================//
    // Generates the primitive group
    // Only single primitive group per type can be available in a tracer
    virtual PrimGroupId     CreatePrimitiveGroup(std::string typeName) = 0;
    virtual PrimBatchId     ReservePrimitiveBatch(PrimGroupId, PrimCount) = 0;
    virtual PrimBatchIdList ReservePrimitiveBatches(PrimGroupId, std::vector<PrimCount>) = 0;
    // Commit (The actual allocation will occur here)
    virtual void            CommitPrimReservations(PrimGroupId) = 0;
    virtual bool            IsPrimCommitted(PrimGroupId) const = 0;
    // Acquire Attribute Info
    virtual void            PushPrimAttribute(PrimGroupId, PrimBatchId,
                                              uint32_t attributeIndex,
                                              TransientData data) = 0;
    virtual void            PushPrimAttribute(PrimGroupId, PrimBatchId,
                                              uint32_t attributeIndex,
                                              Vector2ui subBatchRange,
                                              TransientData data) = 0;
    virtual void            TransformPrimitives(PrimGroupId,
                                                std::vector<PrimBatchId>,
                                                std::vector<Matrix4x4>) = 0;
    //================================//
    //            Material            //
    //================================//
    virtual MatGroupId      CreateMaterialGroup(std::string typeName) = 0;
    virtual MaterialId      ReserveMaterial(MatGroupId, AttributeCountList,
                                            MediumPair = TracerConstants::VacuumMediumPair) = 0;
    virtual MaterialIdList  ReserveMaterials(MatGroupId,
                                             std::vector<AttributeCountList>,
                                             std::vector<MediumPair> = {}) = 0;
    //
    virtual void        CommitMatReservations(MatGroupId) = 0;
    virtual bool        IsMatCommitted(MatGroupId) const = 0;
    //
    virtual void        PushMatAttribute(MatGroupId, CommonIdRange range,
                                         uint32_t attributeIndex,
                                         TransientData data) = 0;
    virtual void        PushMatAttribute(MatGroupId, CommonIdRange range,
                                         uint32_t attributeIndex,
                                         TransientData data,
                                         std::vector<Optional<TextureId>>) = 0;
    virtual void        PushMatAttribute(MatGroupId, CommonIdRange range,
                                         uint32_t attributeIndex,
                                         std::vector<TextureId>) = 0;
    //================================//
    //            Texture             //
    //================================//
    // All textures are implicitly float convertible
    virtual TextureId   CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                        MRayTextureParameters) = 0;
    virtual TextureId   CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                        MRayTextureParameters) = 0;
    virtual void        CommitTextures() = 0;
    // Direct mip data
    // TODO: add more later (sub data etc)
    virtual void        PushTextureData(TextureId, uint32_t mipLevel,
                                        TransientData data) = 0;
    //================================//
    //          Transform             //
    //================================//
    virtual TransGroupId    CreateTransformGroup(std::string typeName) = 0;
    virtual TransformId     ReserveTransformation(TransGroupId, AttributeCountList) = 0;
    virtual TransformIdList ReserveTransformations(TransGroupId, std::vector<AttributeCountList>) = 0;
    //
    virtual void            CommitTransReservations(TransGroupId) = 0;
    virtual bool            IsTransCommitted(TransGroupId) const = 0;
    //
    virtual void            PushTransAttribute(TransGroupId, CommonIdRange range,
                                               uint32_t attributeIndex,
                                               TransientData data) = 0;
    //================================//
    //            Lights              //
    //================================//
    // Analytical / Primitive-backed Lights
    virtual LightGroupId    CreateLightGroup(std::string typeName,
                                             PrimGroupId = TracerConstants::EmptyPrimGroupId) = 0;
    virtual LightId         ReserveLight(LightGroupId, AttributeCountList,
                                         PrimBatchId = TracerConstants::EmptyPrimBatchId) = 0;
    virtual LightIdList     ReserveLights(LightGroupId,
                                          std::vector<AttributeCountList>,
                                          std::vector<PrimBatchId> = std::vector<PrimBatchId>{}) = 0;
    //
    virtual void            CommitLightReservations(LightGroupId) = 0;
    virtual bool            IsLightCommitted(LightGroupId) const = 0;
    //
    virtual void            PushLightAttribute(LightGroupId, CommonIdRange range,
                                               uint32_t attributeIndex,
                                               TransientData data) = 0;
    virtual void            PushLightAttribute(LightGroupId, CommonIdRange range,
                                               uint32_t attributeIndex,
                                               TransientData,
                                               std::vector<Optional<TextureId>>) = 0;
    virtual void            PushLightAttribute(LightGroupId, CommonIdRange range,
                                               uint32_t attributeIndex,
                                               std::vector<TextureId>) = 0;

    //================================//
    //           Cameras              //
    //================================//
    virtual CameraGroupId   CreateCameraGroup(std::string typeName) = 0;
    virtual CameraId        ReserveCamera(CameraGroupId, AttributeCountList) = 0;
    virtual CameraIdList    ReserveCameras(CameraGroupId, std::vector<AttributeCountList>) = 0;
    //
    virtual void            CommitCamReservations(CameraGroupId) = 0;
    virtual bool            IsCamCommitted(CameraGroupId) const = 0;
    //
    virtual void            PushCamAttribute(CameraGroupId, CommonIdRange range,
                                             uint32_t attributeIndex,
                                             TransientData data) = 0;
    //================================//
    //            Medium              //
    //================================//
    virtual MediumGroupId   CreateMediumGroup(std::string typeName) = 0;
    virtual MediumId        ReserveMedium(MediumGroupId, AttributeCountList) = 0;
    virtual MediumIdList    ReserveMediums(MediumGroupId, std::vector<AttributeCountList>) = 0;
    //
    virtual void            CommitMediumReservations(MediumGroupId) = 0;
    virtual bool            IsMediumCommitted(MediumGroupId) const = 0;
    //
    virtual void            PushMediumAttribute(MediumGroupId, CommonIdRange range,
                                                uint32_t attributeIndex,
                                                TransientData data) = 0;
    virtual void            PushMediumAttribute(MediumGroupId, CommonIdRange range,
                                                uint32_t attributeIndex,
                                                TransientData,
                                                std::vector<Optional<TextureId>> textures) = 0;
    virtual void            PushMediumAttribute(MediumGroupId, CommonIdRange range,
                                                uint32_t attributeIndex,
                                                std::vector<TextureId> textures) = 0;

    //================================//
    //            Surfaces            //
    //================================//
    // Basic surface
    virtual SurfaceId       CreateSurface(SurfaceParams) = 0;
    // These may not be "surfaces" by nature but user must register them.
    // Renderer will only use the cameras/lights registered here
    // Same goes for other surfaces as well
    // Primitive-backed lights imply accelerator generation
    virtual LightSurfaceId      SetBoundarySurface(LightSurfaceParams) = 0;
    virtual LightSurfaceId      CreateLightSurface(LightSurfaceParams) = 0;
    virtual CamSurfaceId        CreateCameraSurface(CameraSurfaceParams) = 0;
    virtual SurfaceCommitResult CommitSurfaces() = 0;
    virtual CameraTransform     GetCamTransform(CamSurfaceId) const = 0;

    //================================//
    //           Renderers            //
    //================================//
    virtual RendererId  CreateRenderer(std::string typeName) = 0;
    virtual void        DestroyRenderer(RendererId) = 0;
    virtual void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                              TransientData data) = 0;
    //================================//
    //           Rendering            //
    //================================//
    virtual void                SetupRenderEnv(TimelineSemaphore* semaphore,
                                               uint32_t importAlignment,
                                               uint64_t initialAcquireValue) = 0;
    virtual RenderBufferInfo    StartRender(RendererId, CamSurfaceId,
                                            RenderImageParams,
                                            Optional<uint32_t>,
                                            Optional<uint32_t>) = 0;
    virtual void                SetCameraTransform(RendererId, CameraTransform) = 0;
    virtual void                StopRender() = 0;
    // Renderer does a subsection of the img rendering
    // and returns an output
    virtual RendererOutput DoRenderWork() = 0;

    //================================//
    //             Misc.              //
    //================================//
    virtual void    ClearAll() = 0;
    virtual void    Flush() const = 0;

    virtual GPUThreadInitFunction   GetThreadInitFunction() const = 0;
    virtual void                    SetThreadPool(ThreadPool&) = 0;
    virtual size_t                  TotalDeviceMemory() const = 0;
    virtual size_t                  UsedDeviceMemory() const = 0;
    virtual const TracerParameters& Parameters() const = 0;
};

using TracerConstructorArgs = TypePack<const TracerParameters&>;
