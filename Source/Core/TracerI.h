#pragma once

#include <cstdint>

#include "Definitions.h"
#include "Vector.h"
#include "MRayDataType.h"
#include "DataStructures.h"
#include "System.h"

#include "Common/RenderImageStructs.h"

#define MRAY_GENERIC_ID(NAME, TYPE) enum class NAME : TYPE {}

class TimelineSemaphore;

using GPUThreadInitFunction = void(*)();
namespace BS { class thread_pool; }

struct RenderImageParams
{
    Vector2ui               resolution;
    Vector2ui               regionMin;
    Vector2ui               regionMax;
    //
    TimelineSemaphore*      semaphore;
    uint64_t                initialSemCounter;
};

namespace TransientPoolDetail { class TransientData; }
using TransientData = TransientPoolDetail::TransientData;

enum class PrimitiveAttributeLogic : uint8_t
{
    POSITION,
    INDEX,
    NORMAL,
    RADIUS,
    TANGENT,
    BITANGENT,
    UV0,
    UV1,
    WEIGHT,
    WEIGHT_INDEX,

    END
};

namespace TracerConstants
{
    // This is utilized by static vectors to evade heap,
    // this is currently "small" 65K x 65K textures are max
    static constexpr size_t MaxTextureMipCount = 16;
    static constexpr size_t MaxPrimBatchPerSurface = 8;
    static constexpr size_t MaxAttributePerGroup = 16;
    static constexpr size_t MaxRendererAttributeCount = 32;

    static constexpr std::string_view EmptyPrimName     = "(P)Empty";
    static constexpr std::string_view LIGHT_PREFIX      = "(L)";
    static constexpr std::string_view TRANSFORM_PREFIX  = "(T)";
    static constexpr std::string_view PRIM_PREFIX       = "(P)";
    static constexpr std::string_view MAT_PREFIX        = "(Mt)";
    static constexpr std::string_view CAM_PREFIX        = "(C)";
    static constexpr std::string_view MEDIUM_PREFIX     = "(Md)";
    static constexpr std::string_view ACCEL_PREFIX      = "(A)";
    static constexpr std::string_view RENDERER_PREFIX   = "(R)";
}

// Accelerators are responsible for accelerating ray/surface interactions
// This is abstracted away but exposed to the user for prototyping different
// accelerators. This is less usefull due to hw-accelerated ray tracing.
//
// The old design supported mixing and matching "Base Accelerator"
// (TLAS, IAS on other APIs) with bottom-level accelerators. This
// design only enables the user to set a single accelerator type
// for the entire scene. Hardware acceleration APIs, such asOptiX,
// did not support it anyway so it is removed. Internally software stack
// utilizes the old implementation for simulating two-level acceleration
// hierarchy.
//
// Currently Accepted types are
//
//  -- SOFTWARE_NONE :  No acceleration, ray caster does a !LINEAR SEARCH! over the
//                      primitives (Should not be used, it is for debugging, testing)
//  -- SOFTWARE_BVH  :  Very basic midpoint BVH. Provided for completeness sake and
//                      should not be used.
//  -- HARDWARE      :  On CUDA, it utilizes OptiX for hardware acceleration.
enum class AcceleratorType : uint8_t
{
    SOFTWARE_NONE,
    SOFTWARE_BASIC_BVH,
    HARDWARE
};

enum class SamplerType : uint8_t
{
    INDEPENDENT
};

struct TracerParameters
{
    // Random Seed value, many samplers etc.
    // will start via this seed if applicable
    uint64_t        seed = 0;
    // Accelerator move, software/hardware
    AcceleratorType accelMode = AcceleratorType::HARDWARE;
    // Item pool size, amount of "items" (paths/rays/etc) processed
    // in parallel
    uint32_t        itemPoolSize = 1 << 21; // 2^21 ~= 2M
    // Current sampler logic,
    SamplerType     samplerType = SamplerType::INDEPENDENT;
    // Texture Related
    uint32_t            clampedTexRes = std::numeric_limits<uint32_t>::max();
    MRayColorSpaceEnum  globalTextureColorSpace = MRayColorSpaceEnum::MR_ACES_CG;
};

enum class AttributeOptionality : uint8_t
{
    MR_MANDATORY,
    MR_OPTIONAL
};

enum class AttributeTexturable : uint8_t
{
    MR_CONSTANT_ONLY,
    MR_TEXTURE_OR_CONSTANT,
    MR_TEXTURE_ONLY
};

enum class AttributeIsColor : uint8_t
{
    IS_COLOR,
    IS_PURE_DATA
};

enum class AttributeIsArray : uint8_t
{
    IS_SCALAR,
    IS_ARRAY
};

// Generic Texture Input Parameters
struct MRayTextureParameters
{
    MRayPixelTypeRT             pixelType;
    MRayColorSpaceEnum          colorSpace = MRayColorSpaceEnum::MR_DEFAULT;
    AttributeIsColor            isColor = AttributeIsColor::IS_COLOR;
    MRayTextureEdgeResolveEnum  edgeResolve = MRayTextureEdgeResolveEnum::MR_WRAP;
    MRayTextureInterpEnum       interpolation = MRayTextureInterpEnum::MR_NEAREST;
};

// Generic Attribute Info
struct GenericAttributeInfo : public Tuple<std::string, MRayDataTypeRT,
                                           AttributeIsArray, AttributeOptionality>
{
    using Base = Tuple<std::string, MRayDataTypeRT,
                       AttributeIsArray, AttributeOptionality>;
    using Base::Base;
    enum E
    {
        LOGIC_INDEX         = 0,
        LAYOUT_INDEX        = 1,
        IS_ARRAY_INDEX      = 2,
        OPTIONALITY_INDEX   = 3
    };
};

struct TexturedAttributeInfo : public Tuple<std::string, MRayDataTypeRT,
                                            AttributeIsArray, AttributeOptionality,
                                            AttributeTexturable, AttributeIsColor>
{
    using Base = Tuple<std::string, MRayDataTypeRT,
                       AttributeIsArray, AttributeOptionality,
                       AttributeTexturable, AttributeIsColor>;
    using Base::Base;
    enum E
    {
        LOGIC_INDEX         = 0,
        LAYOUT_INDEX        = 1,
        IS_ARRAY_INDEX      = 2,
        OPTIONALITY_INDEX   = 3,
        TEXTURABLE_INDEX    = 4,
        COLOROMETRY_INDEX   = 5,
    };
};

using GenericAttributeInfoList = StaticVector<GenericAttributeInfo,
                                              TracerConstants::MaxAttributePerGroup>;
using TexturedAttributeInfoList = StaticVector<TexturedAttributeInfo,
                                               TracerConstants::MaxAttributePerGroup>;
using TypeNameList = std::vector<std::string_view>;

struct PrimAttributeStringifier
{
    using enum PrimitiveAttributeLogic;
    static constexpr std::array<std::string_view, static_cast<size_t>(END)> Names =
    {
        "Position",
        "Index",
        "Normal",
        "Radius",
        "Tangent",
        "BiTangent",
        "UV0",
        "UV1",
        "Weight",
        "Weight Index"
    };
    static constexpr std::string_view           ToString(PrimitiveAttributeLogic e);
    static constexpr PrimitiveAttributeLogic    FromString(std::string_view e);
};

constexpr std::string_view PrimAttributeStringifier::ToString(PrimitiveAttributeLogic e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr PrimitiveAttributeLogic PrimAttributeStringifier::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<PrimitiveAttributeLogic>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return PrimitiveAttributeLogic(i);
        i++;
    }
    return PrimitiveAttributeLogic(END);
}

// For surface commit analytic information
struct SurfaceCommitResult
{
    AABB3 aabb;
    size_t instanceCount;
    size_t acceleratorCount;
};

// Prim related
MRAY_GENERIC_ID(PrimGroupId, uint32_t);
MRAY_GENERIC_ID(PrimBatchId, uint32_t);
struct PrimCount { uint32_t primCount; uint32_t attributeCount; };
using PrimBatchIdList = std::vector<PrimBatchId>;
struct PrimAttributeInfo : public Tuple<PrimitiveAttributeLogic, MRayDataTypeRT,
                                        AttributeIsArray, AttributeOptionality>
{
    using Base = Tuple<PrimitiveAttributeLogic, MRayDataTypeRT,
                       AttributeIsArray, AttributeOptionality>;
    using Base::Base;
    enum
    {
        LOGIC_INDEX         = 0,
        LAYOUT_INDEX        = 1,
        IS_ARRAY_INDEX      = 2,
        OPTIONALITY_INDEX   = 3
    };
};
using PrimAttributeInfoList = StaticVector<PrimAttributeInfo,
                                           TracerConstants::MaxAttributePerGroup>;
// Texture Related
MRAY_GENERIC_ID(TextureId, uint32_t);
// Transform Related
MRAY_GENERIC_ID(TransGroupId, uint32_t);
MRAY_GENERIC_ID(TransformId, uint32_t);
using TransAttributeInfo = GenericAttributeInfo;
using TransAttributeInfoList = GenericAttributeInfoList;
// Light Related
MRAY_GENERIC_ID(LightGroupId, uint32_t);
MRAY_GENERIC_ID(LightId, uint32_t);
using LightAttributeInfo = TexturedAttributeInfo;
using LightAttributeInfoList = TexturedAttributeInfoList;
// Camera Related
MRAY_GENERIC_ID(CameraGroupId, uint32_t);
MRAY_GENERIC_ID(CameraId, uint32_t);
using CamAttributeInfo = GenericAttributeInfo;
using CamAttributeInfoList = GenericAttributeInfoList;
// Material Related
MRAY_GENERIC_ID(MatGroupId, uint32_t);
MRAY_GENERIC_ID(MaterialId, uint32_t);
using MatAttributeInfo = TexturedAttributeInfo;
using MatAttributeInfoList = TexturedAttributeInfoList;
// Medium Related
MRAY_GENERIC_ID(MediumGroupId, uint32_t);
MRAY_GENERIC_ID(MediumId, uint32_t);
using MediumPair = Pair<MediumId, MediumId>;
using MediumAttributeInfo = TexturedAttributeInfo;
using MediumAttributeInfoList = TexturedAttributeInfoList;
// Surface Related
MRAY_GENERIC_ID(SurfaceId, uint32_t);
MRAY_GENERIC_ID(LightSurfaceId, uint32_t);
MRAY_GENERIC_ID(CamSurfaceId, uint32_t);
using SurfaceMatList        = StaticVector<MaterialId, TracerConstants::MaxPrimBatchPerSurface>;
using SurfacePrimList       = StaticVector<PrimBatchId, TracerConstants::MaxPrimBatchPerSurface>;
using OptionalAlphaMapList  = StaticVector<Optional<TextureId>, TracerConstants::MaxPrimBatchPerSurface>;
using CullBackfaceFlagList  = StaticVector<bool, TracerConstants::MaxPrimBatchPerSurface>;
// Renderer Related
MRAY_GENERIC_ID(RendererId, uint32_t);
using RendererAttributeInfo = GenericAttributeInfo;
using RendererAttributeInfoList = StaticVector<GenericAttributeInfo,
                                               TracerConstants::MaxRendererAttributeCount>;

using MaterialIdList    = std::vector<MaterialId>;
using TransformIdList   = std::vector<TransformId>;
using MediumIdList      = std::vector<MediumId>;
using LightIdList       = std::vector<LightId>;
using CameraIdList      = std::vector<CameraId>;

using AttributeCountList = StaticVector<size_t, TracerConstants::MaxAttributePerGroup>;

// For transfer of options
struct RendererOptionPack
{
    using AttributeList = StaticVector<TransientData,
        TracerConstants::MaxRendererAttributeCount>;
    //
    RendererAttributeInfoList   paramTypes;
    AttributeList               attributes;
};

namespace TracerConstants
{
    // Implicit Mediums that are always present on a tracer system
    static constexpr MediumId VacuumMediumId            = MediumId(0);
    static constexpr LightId NullLightId                = LightId(0);
    static constexpr TransformId IdentityTransformId    = TransformId(0);
    static constexpr PrimGroupId EmptyPrimitive         = PrimGroupId{0};
    static constexpr PrimBatchId EmptyPrimBatch         = PrimBatchId{0};
    static constexpr TextureId InvalidTexture           = TextureId{0};
    static constexpr MediumPair VacuumMediumPair        = std::make_pair(VacuumMediumId, VacuumMediumId);

    static const auto NoAlphaMapList = OptionalAlphaMapList(StaticVecSize(MaxPrimBatchPerSurface),
                                                            std::nullopt);
    static const auto CullFaceTrueList = CullBackfaceFlagList(StaticVecSize(MaxPrimBatchPerSurface),
                                                              true);
};

struct SurfaceParams
{
    SurfacePrimList         primBatches;
    SurfaceMatList          materials;
    TransformId             transformId     = TracerConstants::IdentityTransformId;
    OptionalAlphaMapList    alphaMaps       = TracerConstants::NoAlphaMapList;
    CullBackfaceFlagList    cullFaceFlags   = TracerConstants::CullFaceTrueList;
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
    virtual void        PushMatAttribute(MatGroupId, Vector2ui range,
                                         uint32_t attributeIndex,
                                         TransientData data) = 0;
    virtual void        PushMatAttribute(MatGroupId, Vector2ui range,
                                         uint32_t attributeIndex,
                                         TransientData data,
                                         std::vector<Optional<TextureId>>) = 0;
    virtual void        PushMatAttribute(MatGroupId, Vector2ui range,
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
    virtual void            PushTransAttribute(TransGroupId, Vector2ui range,
                                               uint32_t attributeIndex,
                                               TransientData data) = 0;
    //================================//
    //            Lights              //
    //================================//
    // Analytical / Primitive-backed Lights
    virtual LightGroupId    CreateLightGroup(std::string typeName,
                                             PrimGroupId = TracerConstants::EmptyPrimitive) = 0;
    virtual LightId         ReserveLight(LightGroupId, AttributeCountList,
                                         PrimBatchId = TracerConstants::EmptyPrimBatch) = 0;
    virtual LightIdList     ReserveLights(LightGroupId,
                                          std::vector<AttributeCountList>,
                                          std::vector<PrimBatchId> = std::vector<PrimBatchId>{}) = 0;
    //
    virtual void            CommitLightReservations(LightGroupId) = 0;
    virtual bool            IsLightCommitted(LightGroupId) const = 0;
    //
    virtual void            PushLightAttribute(LightGroupId, Vector2ui range,
                                               uint32_t attributeIndex,
                                               TransientData data) = 0;
    virtual void            PushLightAttribute(LightGroupId, Vector2ui range,
                                               uint32_t attributeIndex,
                                               TransientData,
                                               std::vector<Optional<TextureId>>) = 0;
    virtual void            PushLightAttribute(LightGroupId, Vector2ui range,
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
    virtual void            PushCamAttribute(CameraGroupId, Vector2ui range,
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
    virtual void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                                uint32_t attributeIndex,
                                                TransientData data) = 0;
    virtual void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                                uint32_t attributeIndex,
                                                TransientData,
                                                std::vector<Optional<TextureId>> textures) = 0;
    virtual void            PushMediumAttribute(MediumGroupId, Vector2ui range,
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
    virtual RenderBufferInfo    StartRender(RendererId, CamSurfaceId,
                                            RenderImageParams,
                                            Optional<uint32_t>,
                                            Optional<uint32_t>,
                                            Optional<CameraTransform>) = 0;
    virtual void                StopRender() = 0;
    // Renderer does a subsection of the img rendering
    // and returns an output
    virtual RendererOutput DoRenderWork() = 0;

    //================================//
    //             Misc.              //
    //================================//
    virtual void    ClearAll() = 0;

    virtual GPUThreadInitFunction   GetThreadInitFunction() const = 0;
    virtual void                    SetThreadPool(BS::thread_pool&) = 0;
    virtual size_t                  TotalDeviceMemory() const = 0;
    virtual size_t                  UsedDeviceMemory() const = 0;
    virtual const TracerParameters& Parameters() const = 0;
};

using TracerConstructorArgs = Tuple<const TracerParameters&>;

// formatter for AcceleratorType
inline std::string_view format_as(AcceleratorType t)
{
    using namespace std::string_view_literals;
    using enum AcceleratorType;
    switch(t)
    {
        case SOFTWARE_NONE:         return "SOFTWARE_LINEAR"sv;
        case SOFTWARE_BASIC_BVH:    return "SOFTWARE_BVH"sv;
        case HARDWARE:              return "HARDWARE"sv;
        default:                    return "UNKNOWN"sv;
    }
}