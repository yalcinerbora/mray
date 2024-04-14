#pragma once

#include <cstdint>

#include "Definitions.h"
#include "Vector.h"
#include "MRayDataType.h"
#include "DataStructures.h"
#include "System.h"

#define MRAY_GENERIC_ID(NAME, TYPE) enum class NAME : TYPE {}

struct TracerImgOutput
{
    SystemSemaphoreHandle   imgReadySignal;
    SystemSemaphoreHandle   imgUsedSignal;
    const Byte*             imagePtr;
    uint32_t                sampleSectionOffset;
    Vector2i                regionMin;
    Vector2i                regionMax;
    uint32_t                layerIndex;
    MRayColorSpaceEnum      colorSpace;
};

struct RenderImageParams
{
    Vector2i            resolution;
    Vector2i            regionMin;
    Vector2i            regionMax;
    MRayColorSpaceEnum  colorSpace;
};

namespace TransientPoolDetail { class TransientData; }
using TransientData = TransientPoolDetail::TransientData;

enum class PrimitiveAttributeLogic
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
    static constexpr size_t MaxPrimBatchPerSurface = 8;
    static constexpr size_t MaxAttributePerGroup = 16;

    static constexpr std::string_view LIGHT_PREFIX      = "(L)";
    static constexpr std::string_view TRANSFORM_PREFIX  = "(T)";
    static constexpr std::string_view PRIM_PREFIX       = "(P)";
    static constexpr std::string_view MAT_PREFIX        = "(Mt)";
    static constexpr std::string_view CAM_PREFIX        = "(C)";
    static constexpr std::string_view MEDIUM_PREFIX     = "(Md)";
}

enum class AcceleratorType
{
    SOFTWARE_NONE,
    SOFTWARE_BASIC_BVH,
    HARDWARE
};

enum class AttributeOptionality
{
    MR_MANDATORY,
    MR_OPTIONAL
};

enum class AttributeTexturable
{
    MR_CONSTANT_ONLY,
    MR_TEXTURE_OR_CONSTANT,
    MR_TEXTURE_ONLY
};

enum class AttributeIsColor
{
    IS_COLOR,
    IS_PURE_DATA
};

enum class AttributeIsArray
{
    IS_SCALAR,
    IS_ARRAY
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

using TypeNameList = std::vector<std::string>;

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
using RendererAttributeInfoList = GenericAttributeInfoList;

using MaterialIdList    = std::vector<MaterialId>;
using TransformIdList   = std::vector<TransformId>;
using MediumIdList      = std::vector<MediumId>;
using LightIdList       = std::vector<LightId>;
using CameraIdList      = std::vector<CameraId>;

using AttributeCountList = StaticVector<size_t, TracerConstants::MaxAttributePerGroup>;

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
    // All textures must be defined on the same color space
    virtual void        CommitTexColorSpace(MRayColorSpaceEnum = MRayColorSpaceEnum::MR_DEFAULT) = 0;
    // All textures are implicitly float convertible
    virtual TextureId   CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                        MRayPixelEnum pixelType,
                                        AttributeIsColor isColorTexture) = 0;
    virtual TextureId   CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                        MRayPixelEnum pixelType,
                                        AttributeIsColor isColorTexture) = 0;
    // Requested texture may be represented by a different type
    // (float3 -> float4 due to padding)
    virtual MRayDataTypeRT  GetTexturePixelType(TextureId) const = 0;
    //
    virtual void            CommitTextures() = 0;
    // Direct mip data
    // TODO: add more later (sub data etc)
    virtual void            PushTextureData(TextureId, uint32_t mipLevel,
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
    //     Accelerator & Surfaces     //
    //================================//
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
    //
    // Surfaces
    // Basic surface
    virtual SurfaceId       CreateSurface(SurfacePrimList primBatches,
                                          SurfaceMatList material,
                                          TransformId = TracerConstants::IdentityTransformId,
                                          OptionalAlphaMapList alphaMaps = TracerConstants::NoAlphaMapList,
                                          CullBackfaceFlagList cullFaceFlags = TracerConstants::CullFaceTrueList) = 0;
    // These may not be "surfaces" by nature but user must register them.
    // Renderer will only use the cameras/lights registered here
    // Same goes for other surfaces as well
    // Primitive-backed lights imply accelerator generation
    virtual LightSurfaceId  CreateLightSurface(LightId,
                                               TransformId = TracerConstants::IdentityTransformId,
                                               MediumId = TracerConstants::VacuumMediumId) = 0;
    virtual CamSurfaceId    CreateCameraSurface(CameraId,
                                                TransformId = TracerConstants::IdentityTransformId,
                                                MediumId = TracerConstants::VacuumMediumId) = 0;

    virtual void            CommitSurfaces(AcceleratorType) = 0;

    //================================//
    //           Renderers            //
    //================================//
    virtual RendererId  CreateRenderer(std::string typeName) = 0;
    virtual void        DestroyRenderer(RendererId) = 0;
    //
    virtual void        CommitRendererReservations(RendererId) = 0;
    virtual bool        IsRendererCommitted(RendererId) const = 0;
    virtual void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                              TransientData data) = 0;
    //================================//
    //           Rendering            //
    //================================//
    virtual void        StartRender(RendererId, CamSurfaceId,
                                    RenderImageParams) = 0;
    virtual void        StopRender() = 0;

    // Renderer does a subsection of the img rendering
    // and returns an output
    virtual Optional<TracerImgOutput> DoRenderWork() = 0;

};