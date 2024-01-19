#pragma once

#include <future>
#include <cstdint>

#include "Core/Definitions.h"
#include "Core/Vector.h"
#include "Core/MRayDataType.h"

#include "TracerTypes.h"

template <std::unsigned_integral T>
struct GenericId
{
    T id;
    constexpr explicit operator T() { return id; }
};

#define MRAY_GENERIC_ID_BODY(NAME, TYPE) NAME : GenericId<TYPE>\
{\
    using T = TYPE; \
    using GenericId<T>::id; \
    using GenericId<T>::operator T; \
}

namespace TracerConstants
{
    static constexpr size_t MaxPrimBatchPerSurface = 8;
}

// Generic Attribute Info
using GenericAttributeInfo = std::pair<std::string, MRayDataTypeRT>;
using TypeNameList = std::vector<std::string>;

// Prim related
struct MRAY_GENERIC_ID_BODY(PrimGroupId, uint32_t);
struct MRAY_GENERIC_ID_BODY(PrimLocalBatchId, uint32_t);
struct PrimBatchId { PrimGroupId primGroupId; PrimLocalBatchId localBatchId; };
struct PrimCount { uint32_t primCount; uint32_t attributeCount; };
using PrimBatchIdList = std::vector<PrimBatchId>;
using PrimAttributeInfo = GenericAttributeInfo;
using PrimAttributeInfoList = std::vector<PrimAttributeInfo>;
// Texture Related
struct MRAY_GENERIC_ID_BODY(TextureId, uint32_t);
// Transform Related
struct MRAY_GENERIC_ID_BODY(TransGroupId, uint32_t);
using TransAttributeInfo = GenericAttributeInfo;
using TransAttributeInfoList = std::vector<TransAttributeInfo>;
// Light Related
struct MRAY_GENERIC_ID_BODY(LightGroupId, uint32_t);
using LightAttributeInfo = GenericAttributeInfo;
using LightAttributeInfoList = std::vector<LightAttributeInfo>;
// Camera Related
struct MRAY_GENERIC_ID_BODY(CameraGroupId, uint32_t);
using CamAttributeInfo = GenericAttributeInfo;
using CamAttributeInfoList = std::vector<CamAttributeInfo>;
// Material Related
struct MRAY_GENERIC_ID_BODY(MatGroupId, uint32_t);
using MatAttributeInfo = GenericAttributeInfo;
using MatAttributeInfoList = std::vector<MatAttributeInfo>;
// Medium Related
struct MRAY_GENERIC_ID_BODY(MediumGroupId, uint32_t);
using MediumAttributeInfo = GenericAttributeInfo;
using MediumAttributeInfoList = std::vector<MediumAttributeInfo>;
// Surface Related
struct MRAY_GENERIC_ID_BODY(SurfaceId, uint32_t);
struct MRAY_GENERIC_ID_BODY(LightSurfaceId, uint32_t);
struct MRAY_GENERIC_ID_BODY(CamSurfaceId, uint32_t);
using SurfaceMatList = std::array<PrimBatchId, TracerConstants::MaxPrimBatchPerSurface>;
using SurfacePrimList = std::array<PrimLocalBatchId, TracerConstants::MaxPrimBatchPerSurface>;
using OptionalAlphaMapList = std::array<Optional<TextureId>, TracerConstants::MaxPrimBatchPerSurface>;
using CullBackfaceFlagList = std::array<bool, TracerConstants::MaxPrimBatchPerSurface>;
// Renderer Related
struct MRAY_GENERIC_ID_BODY(RendererId, uint32_t);
using RendererAttributeInfo = GenericAttributeInfo;
using RendererAttributeInfoList = std::vector<GenericAttributeInfo>;

namespace TracerConstants
{
    // Implicit Mediums that are always present on a tracer system
    static constexpr MediumId VacuumMediumId        = MediumId(0);
    static constexpr LightId NullLightId            = LightId(0);
    static constexpr LightId IdentityTransformId    = TransformId(0);
    static constexpr PrimGroupId EmptyPrimitive     = PrimGroupId{0};
    static constexpr PrimBatchId EmptyPrimBatch     = PrimBatchId{EmptyPrimitive, PrimLocalBatchId{0}};

    static constexpr auto NoAlphaMapList = OptionalAlphaMapList
    {
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt
    };

    static constexpr auto CullFaceTrueList = CullBackfaceFlagList
    {
        true, true, true, true,
        true, true, true, true
    };
};

class TracerInterfaceI
{
    public:
    virtual     ~TracerInterfaceI() = default;

    //================================//
    //            Generic             //
    //================================//
    virtual TypeNameList        PrimitiveGroups() const = 0;
    virtual TypeNameList        MaterialGroups() const = 0;
    virtual TypeNameList        TransformGroups() const = 0;
    virtual TypeNameList        CameraGroups() const = 0;
    virtual TypeNameList        MediumGroups() const = 0;

    virtual PrimAttributeInfoList       PrimAttributeInfo(PrimGroupId) const = 0;
    virtual CamAttributeInfoList        CamAttributeInfo(CameraGroupId) const = 0;
    virtual MediumAttributeInfoList     MediumAttributeInfo(MediumGroupId) const = 0;
    virtual MatAttributeInfoList        MatAttributeInfo(MatGroupId) const = 0;
    virtual TransAttributeInfoList      TransAttributeInfo(TransGroupId) const = 0;
    virtual LightAttributeInfoList      LightAttributeInfo(LightGroupId) const = 0;
    virtual RendererAttributeInfoList   RendererAttributeInfo(RendererId) const = 0;

    //================================//
    //           Primitive            //
    //================================//
    // Generates the primitive group
    // Only single primitive group per type can be available in a tracer
    virtual PrimGroupId     CreatePrimitiveGroup(std::string typeName) = 0;
    virtual PrimBatchId     ReservePrimitiveBatch(PrimGroupId, PrimCount) = 0;
    virtual PrimBatchIdList ReservePrimitiveBatches(PrimGroupId,
                                                    std::vector<PrimCount> primitiveCounts) = 0;
    // Commit (The actual allocation will occur here)
    virtual void            CommitPrimReservations(PrimGroupId) = 0;
    virtual bool            IsPrimCommitted(PrimGroupId) const = 0;
    // Acquire Attribute Info
    virtual void            PushPrimAttribute(PrimBatchId,
                                              uint32_t attributeIndex,
                                              std::vector<Byte> data) = 0;
    virtual void            PushPrimAttribute(PrimBatchId,
                                              uint32_t attributeIndex,
                                              Vector2ui subBatchRange,
                                              std::vector<Byte> data) = 0;
    //================================//
    //            Material            //
    //================================//
    virtual MatGroupId  CreateMaterialGroup(std::string typeName) = 0;
    virtual MaterialId  ReserveMaterial(MatGroupId,
                                        MediumId frontMedium = TracerConstants::VacuumMediumId,
                                        MediumId backMedium = TracerConstants::VacuumMediumId) = 0;
    //
    virtual void        CommitMatReservations(MatGroupId) = 0;
    virtual bool        IsMatCommitted(MatGroupId) const = 0;
    //
    virtual void        PushMatAttribute(MatGroupId, Vector2ui range,
                                         uint32_t attributeIndex,
                                         std::vector<Byte> data) = 0;
    virtual void        PushMatAttribute(MatGroupId, Vector2ui range,
                                         uint32_t attributeIndex,
                                         std::vector<TextureId>) = 0;
    //================================//
    //            Texture             //
    //================================//
    // All textures must be defined on the same color space
    virtual void        CommitTexColorSpace(MRayColorSpace = MRayColorSpace::RGB_LINEAR) = 0;
    // All textures are implicitly float convertible
    virtual TextureId   CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                        MRayDataEnum pixelType) = 0;
    virtual TextureId   CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                        MRayDataEnum pixelType) = 0;
    // Requested texture may be represented by a different type
    // (float3 -> float4 due to padding)
    virtual MRayDataTypeRT  GetTexturePixelType(TextureId) const = 0;
    //
    virtual void            CommitTextures() = 0;
    // Direct mip data
    // TODO: add more later (sub data etc)
    virtual void            PushTextureData(TextureId, uint32_t mipLevel,
                                            std::vector<Byte> data) = 0;
    //================================//
    //          Transform             //
    //================================//
    virtual TransGroupId    CreateTransformGroup(std::string typeName) = 0;
    virtual TransformIdList ReserveTransformations(TransGroupId, uint32_t count) = 0;
    //
    virtual void            CommitTransReservations(TransGroupId) = 0;
    virtual bool            IsTransCommitted(TransGroupId) const = 0;
    //
    virtual void            PushTransAttribute(TransGroupId, Vector2ui range,
                                               uint32_t attributeIndex,
                                               std::vector<Byte> data) = 0;
    //================================//
    //            Lights              //
    //================================//
    // Analytical / Primitive-backed Lights
    virtual LightGroupId    CreateLightGroup(std::string typeName,
                                             PrimGroupId = TracerConstants::EmptyPrimitive) = 0;
    virtual LightIdList     ReserveLights(LightGroupId,
                                          PrimBatchId = TracerConstants::EmptyPrimBatch) = 0;
    //
    virtual void            CommitLightReservations(LightGroupId) = 0;
    virtual bool            IsLightCommitted(LightGroupId) const = 0;
    //
    virtual void            PushLightAttribute(LightGroupId, Vector2ui range,
                                               uint32_t attributeIndex,
                                               std::vector<Byte> data) = 0;
    //================================//
    //           Cameras              //
    //================================//
    virtual CameraGroupId   CreateCameraGroup(std::string typeName) = 0;
    virtual CameraIdList    ReserveCameras(CameraGroupId, uint32_t count) = 0;
    //
    virtual void            CommitCamReservations(CameraGroupId) = 0;
    virtual bool            IsCamCommitted(CameraGroupId) const = 0;
    //
    virtual void            PushCamAttribute(CameraGroupId, Vector2ui range,
                                             uint32_t attributeIndex,
                                             std::vector<Byte> data) = 0;
    //================================//
    //            Medium              //
    //================================//
    virtual MediumGroupId   CreateMediumGroup(std::string typeName) = 0;
    virtual MediumIdList    ReserveMediums(MediumGroupId, uint32_t count) = 0;
    //
    virtual void            CommitMediumReservations(MediumGroupId) = 0;
    virtual bool            IsMediumCommitted(MediumGroupId) const = 0;
    //
    virtual void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                                uint32_t attributeIndex,
                                                std::vector<Byte> data) = 0;
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
    //
    virtual void        CommitRendererReservations(RendererId) = 0;
    virtual bool        IsRendererCommitted(RendererId) const = 0;
    virtual void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                              std::vector<Byte> data) = 0;
    //================================//
    //           Rendering            //
    //================================//
    virtual void        StartRender(RendererId, CamSurfaceId) = 0;
    virtual void        StoptRender() = 0;
};