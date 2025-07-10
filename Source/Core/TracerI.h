#pragma once

#include <cstdint>

#include "Definitions.h"
#include "Vector.h"
#include "MRayDataType.h"
#include "DataStructures.h"
#include "MRayDescriptions.h"

#include "Common/RenderImageStructs.h"

#include "TransientPool/TransientPool.h"

#define MRAY_GENERIC_ID(NAME, TYPE) enum class NAME : TYPE {}

using CommonId = MRay::CommonKey;
using CommonIdRange = Vector<2, CommonId>;

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
    // Primitive/Material pairs can be batched and a single
    // accelerator is constructed per batch this is the upper limit
    // of such pairs
    static constexpr size_t MaxPrimBatchPerSurface = 8;
    // Each "group" can have at most N different attributes
    // (i.e., a triangle primitive's "position" "normal" "uv" etc.)
    static constexpr size_t MaxAttributePerGroup = 16;
    // Same as above but for renderer
    static constexpr size_t MaxRendererAttributeCount = 32;
    // Renderer can define at most N work per Mat/Prim/Transform
    // triplet. Most of the time single work definition is enough.
    // but some renderers (surface renderer) may define multiple
    // works, and change the work logic according to input parameters
    static constexpr size_t MaxRenderWorkPerTriplet = 4;
    // Accelerator report hits as barycentric coordinates for triangles,
    // and spherical coordinates for spheres. So two is currently enough.
    // Volume hits will require 3 (local space x, y, z) for hits probably
    // but these can be generated from position.
    // In future, there may be different primitives require more that two
    // hit parametrization (This should rarely be an issue since surfaces
    // are inherently 2D), and this may be changed
    static constexpr size_t MaxHitFloatCount = 2;
    // Maximum camera size, a renderer will allocate this
    // much memory, and camera should fit into this.
    // This will be compile time checked.
    static constexpr size_t MaxCameraInstanceByteSize = 512;

    static constexpr std::string_view IdentityTransName = "(T)Identity";
    static constexpr std::string_view NullLightName     = "(L)Null";
    static constexpr std::string_view EmptyPrimName     = "(P)Empty";
    static constexpr std::string_view VacuumMediumName  = "(Md)Vacuum";

    static constexpr std::string_view LIGHT_PREFIX      = "(L)";
    static constexpr std::string_view TRANSFORM_PREFIX  = "(T)";
    static constexpr std::string_view PRIM_PREFIX       = "(P)";
    static constexpr std::string_view MAT_PREFIX        = "(Mt)";
    static constexpr std::string_view CAM_PREFIX        = "(C)";
    static constexpr std::string_view MEDIUM_PREFIX     = "(Md)";
    static constexpr std::string_view ACCEL_PREFIX      = "(A)";
    static constexpr std::string_view RENDERER_PREFIX   = "(R)";
    static constexpr std::string_view WORK_PREFIX       = "(W)";
}

// Accelerators are responsible for accelerating ray/surface interactions
// This is abstracted away but exposed to the user for prototyping different
// accelerators. This is less useful due to hw-accelerated ray tracing.
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
//                      primitives (Should not be used, it is for debugging,
//                      testing etc.)
//  -- SOFTWARE_BVH  :  Very basic LBVH. Each triangle's center is converted to
//                      a morton code and these are sorted. Provided for completeness
//                      sake and should not be used.
//  -- HARDWARE      :  On CUDA, it utilizes OptiX for hardware acceleration.
class AcceleratorType
{
    public:
    enum E
    {
        SOFTWARE_NONE,
        SOFTWARE_BASIC_BVH,
        HARDWARE,

        END
    };

    private:
    static constexpr std::array<std::string_view, static_cast<size_t>(END)> Names =
    {
        "Linear",
        "BVH",
        "Hardware"
    };

    public:
    E type;

    // We use this on a map, so overload less
    bool operator<(AcceleratorType t) const;

    static constexpr std::string_view   ToString(E e);
    static constexpr E                  FromString(std::string_view e);
};

class SamplerType
{
    public:
    enum E
    {
        INDEPENDENT,

        END
    };

    private:
    static constexpr std::array<std::string_view, static_cast<size_t>(END)> Names =
    {
        "Independent"
    };

    public:
    E type;

    static constexpr std::string_view   ToString(E e);
    static constexpr E                  FromString(std::string_view e);
};

class FilterType
{
    public:
    enum E
    {
        BOX,
        TENT,
        GAUSSIAN,
        MITCHELL_NETRAVALI,

        END
    };

    private:
    static constexpr std::array<std::string_view, static_cast<size_t>(END)> Names =
    {
        "Box",
        "Tent",
        "Gaussian",
        "Mitchell-Netravali"
    };

    public:
    E       type;
    Float   radius;

    static constexpr auto TYPE_NAME = "type";
    static constexpr auto RADIUS_NAME = "radius";

    static constexpr std::string_view   ToString(E e);
    static constexpr E                  FromString(std::string_view e);
};

struct TracerParameters
{
    // Random Seed value, many samplers etc.
    // will start via this seed if applicable
    uint64_t        seed = 0;
    // Accelerator mode, software/hardware etc.
    AcceleratorType accelMode = AcceleratorType{ AcceleratorType::HARDWARE };
    // Item pool size, amount of "items" (paths/rays/etc) processed
    // in parallel
    uint32_t    parallelizationHint = 1 << 21; // 2^21 ~= 2M
    // Current sampler logic,
    SamplerType samplerType = SamplerType{ SamplerType::INDEPENDENT };
    // Texture Related
    uint32_t            clampedTexRes = std::numeric_limits<uint32_t>::max();
    bool                genMips = false;
    FilterType          mipGenFilter = FilterType{ FilterType::GAUSSIAN, 2.0f };
    MRayColorSpaceEnum  globalTextureColorSpace = MRayColorSpaceEnum::MR_ACES_CG;

    // Film Related
    FilterType          filmFilter = FilterType{FilterType::GAUSSIAN, 1.0f};
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
    MRayColorSpaceEnum          colorSpace      = MRayColorSpaceEnum::MR_DEFAULT;
    Float                       gamma           = Float(1);
    bool                        ignoreResClamp  = false;
    AttributeIsColor            isColor         = AttributeIsColor::IS_COLOR;
    MRayTextureEdgeResolveEnum  edgeResolve     = MRayTextureEdgeResolveEnum::MR_WRAP;
    MRayTextureInterpEnum       interpolation   = MRayTextureInterpEnum::MR_LINEAR;
    MRayTextureReadMode         readMode        = MRayTextureReadMode::MR_PASSTHROUGH;
};

// Generic Attribute Info
struct GenericAttributeInfo : public std::tuple<std::string, MRayDataTypeRT,
                                                AttributeIsArray, AttributeOptionality>
{
    using Base = std::tuple<std::string, MRayDataTypeRT,
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

struct TexturedAttributeInfo : public std::tuple<std::string, MRayDataTypeRT,
                                                 AttributeIsArray, AttributeOptionality,
                                                 AttributeTexturable, AttributeIsColor>
{
    using Base = std::tuple<std::string, MRayDataTypeRT,
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
        COLORIMETRY_INDEX   = 5,
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

// For surface commit analytic information
struct SurfaceCommitResult
{
    AABB3 aabb;
    size_t instanceCount;
    size_t acceleratorCount;
};

// Prim related
MRAY_GENERIC_ID(PrimGroupId, CommonId);
MRAY_GENERIC_ID(PrimBatchId, CommonId);
struct PrimCount { uint32_t primCount; uint32_t attributeCount; };
using PrimBatchIdList = std::vector<PrimBatchId>;
struct PrimAttributeInfo : public std::tuple<PrimitiveAttributeLogic, MRayDataTypeRT,
                                             AttributeIsArray, AttributeOptionality>
{
    using Base = std::tuple<PrimitiveAttributeLogic, MRayDataTypeRT,
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
MRAY_GENERIC_ID(TextureId, CommonId);
// Transform Related
MRAY_GENERIC_ID(TransGroupId, CommonId);
MRAY_GENERIC_ID(TransformId, CommonId);
using TransAttributeInfo = GenericAttributeInfo;
using TransAttributeInfoList = GenericAttributeInfoList;
// Light Related
MRAY_GENERIC_ID(LightGroupId, CommonId);
MRAY_GENERIC_ID(LightId, CommonId);
using LightAttributeInfo = TexturedAttributeInfo;
using LightAttributeInfoList = TexturedAttributeInfoList;
// Camera Related
MRAY_GENERIC_ID(CameraGroupId, CommonId);
MRAY_GENERIC_ID(CameraId, CommonId);
using CamAttributeInfo = GenericAttributeInfo;
using CamAttributeInfoList = GenericAttributeInfoList;
// Material Related
MRAY_GENERIC_ID(MatGroupId, CommonId);
MRAY_GENERIC_ID(MaterialId, CommonId);
using MatAttributeInfo = TexturedAttributeInfo;
using MatAttributeInfoList = TexturedAttributeInfoList;
// Medium Related
MRAY_GENERIC_ID(MediumGroupId, CommonId);
MRAY_GENERIC_ID(MediumId, CommonId);
using MediumPair = Pair<MediumId, MediumId>;
using MediumAttributeInfo = TexturedAttributeInfo;
using MediumAttributeInfoList = TexturedAttributeInfoList;
// Surface Related
MRAY_GENERIC_ID(SurfaceId, CommonId);
MRAY_GENERIC_ID(LightSurfaceId, CommonId);
MRAY_GENERIC_ID(CamSurfaceId, CommonId);
using SurfaceMatList        = StaticVector<MaterialId, TracerConstants::MaxPrimBatchPerSurface>;
using SurfacePrimList       = StaticVector<PrimBatchId, TracerConstants::MaxPrimBatchPerSurface>;
using OptionalAlphaMapList  = StaticVector<Optional<TextureId>, TracerConstants::MaxPrimBatchPerSurface>;
using CullBackfaceFlagList  = StaticVector<bool, TracerConstants::MaxPrimBatchPerSurface>;
// Renderer Related
MRAY_GENERIC_ID(RendererId, CommonId);
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
    static constexpr MediumGroupId VacuumMediumGroupId  = MediumGroupId(0);
    static constexpr MediumId VacuumMediumId            = MediumId(0);

    static constexpr LightGroupId NullLightGroupId      = LightGroupId(0);
    static constexpr LightId NullLightId                = LightId(0);

    static constexpr TransGroupId IdentityTransGroupId  = TransGroupId(0);
    static constexpr TransformId IdentityTransformId    = TransformId(0);

    static constexpr PrimGroupId EmptyPrimGroupId       = PrimGroupId{0};
    static constexpr PrimBatchId EmptyPrimBatchId       = PrimBatchId{0};

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

using TracerConstructorArgs = PackedTypes<const TracerParameters&>;

// We use this on a map, so overload less
inline bool AcceleratorType::operator<(AcceleratorType t) const
{
    return type < t.type;
}

constexpr std::string_view AcceleratorType::ToString(AcceleratorType::E e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename AcceleratorType::E
AcceleratorType::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename AcceleratorType::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return AcceleratorType::E(i);
        i++;
    }
    return END;
}

constexpr std::string_view SamplerType::ToString(typename SamplerType::E e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename SamplerType::E
SamplerType::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename SamplerType::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return SamplerType::E(i);
        i++;
    }
    return END;
}

constexpr std::string_view FilterType::ToString(typename FilterType::E e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename FilterType::E
FilterType::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename FilterType::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return FilterType::E(i);
        i++;
    }
    return END;
}

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
    return PrimitiveAttributeLogic::END;
}

// formatter for AcceleratorType
inline std::string_view format_as(AcceleratorType t)
{
    using namespace std::string_view_literals;
    using enum AcceleratorType::E;
    switch(t.type)
    {
        case SOFTWARE_NONE:         return "SOFTWARE_LINEAR"sv;
        case SOFTWARE_BASIC_BVH:    return "SOFTWARE_BVH"sv;
        case HARDWARE:              return "HARDWARE"sv;
        default:                    return "UNKNOWN"sv;
    }
}