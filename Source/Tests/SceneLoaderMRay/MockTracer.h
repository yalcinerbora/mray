#pragma once

#include "Core/TracerI.h"
#include "MRayInput/MRayInput.h"

#include <map>

class TracerMock : public TracerI
{
    private:
    std::map<PrimGroupId, PrimAttributeInfoList> primAttribMap;

    public:
                        TracerMock();

    TypeNameList        PrimitiveGroups() const override;
    TypeNameList        MaterialGroups() const override;
    TypeNameList        TransformGroups() const override;
    TypeNameList        CameraGroups() const override;
    TypeNameList        MediumGroups() const override;

    PrimAttributeInfoList       AttributeInfo(PrimGroupId) const override;
    CamAttributeInfoList        AttributeInfo(CameraGroupId) const override;
    MediumAttributeInfoList     AttributeInfo(MediumGroupId) const override;
    MatAttributeInfoList        AttributeInfo(MatGroupId) const override;
    TransAttributeInfoList      AttributeInfo(TransGroupId) const override;
    LightAttributeInfoList      AttributeInfo(LightGroupId) const override;
    RendererAttributeInfoList   AttributeInfo(RendererId) const override;

    PrimAttributeInfoList       PrimAttributeInfo(std::string_view) const override;
    CamAttributeInfoList        CamAttributeInfo(std::string_view) const override;
    MediumAttributeInfoList     MediumAttributeInfo(std::string_view) const override;
    MatAttributeInfoList        MatAttributeInfo(std::string_view) const override;
    TransAttributeInfoList      TransAttributeInfo(std::string_view) const override;
    LightAttributeInfoList      LightAttributeInfo(std::string_view) const override;
    RendererAttributeInfoList   RendererAttributeInfo(std::string_view) const override;

    std::string     TypeName(PrimGroupId) const override;
    std::string     TypeName(CameraGroupId) const override;
    std::string     TypeName(MediumGroupId) const override;
    std::string     TypeName(MatGroupId) const override;
    std::string     TypeName(TransGroupId) const override;
    std::string     TypeName(LightGroupId) const override;
    std::string     TypeName(RendererId) const override;

    PrimGroupId     CreatePrimitiveGroup(std::string typeName) override;
    PrimBatchId     ReservePrimitiveBatch(PrimGroupId, PrimCount) override;
    PrimBatchIdList ReservePrimitiveBatches(PrimGroupId,
                                            std::vector<PrimCount> primitiveCounts) override;
    void            CommitPrimReservations(PrimGroupId) override;
    bool            IsPrimCommitted(PrimGroupId) const override;
    void            PushPrimAttribute(PrimBatchId,
                                      uint32_t attributeIndex,
                                      MRayInput data) override;
    void            PushPrimAttribute(PrimBatchId,
                                      uint32_t attributeIndex,
                                      Vector2ui subBatchRange,
                                      MRayInput data) override;

    MatGroupId  CreateMaterialGroup(std::string typeName) override;
    MaterialId  ReserveMaterial(MatGroupId,
                                MediumId frontMedium = TracerConstants::VacuumMediumId,
                                MediumId backMedium = TracerConstants::VacuumMediumId) override;
    void        CommitMatReservations(MatGroupId) override;
    bool        IsMatCommitted(MatGroupId) const override;
    void        PushMatAttribute(MatGroupId, Vector2ui range,
                                 uint32_t attributeIndex,
                                 MRayInput data) override;
    void        PushMatAttribute(MatGroupId, Vector2ui range,
                                 uint32_t attributeIndex,
                                 std::vector<TextureId>) override;


    void            CommitTexColorSpace(MRayColorSpace = MRayColorSpace::RGB_LINEAR) override;
    TextureId       CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                    MRayDataEnum pixelType) override;
    TextureId       CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                    MRayDataEnum pixelType) override;
    MRayDataTypeRT  GetTexturePixelType(TextureId) const override;
    void            CommitTextures() override;
    void            PushTextureData(TextureId, uint32_t mipLevel,
                                    MRayInput data) override;


    TransGroupId    CreateTransformGroup(std::string typeName) override;
    TransformIdList ReserveTransformations(TransGroupId, uint32_t count) override;
    void            CommitTransReservations(TransGroupId) override;
    bool            IsTransCommitted(TransGroupId) const override;
    void            PushTransAttribute(TransGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       MRayInput data) override;


    LightGroupId    CreateLightGroup(std::string typeName,
                                     PrimGroupId = TracerConstants::EmptyPrimitive) override;
    LightIdList     ReserveLights(LightGroupId,
                                  PrimBatchId = TracerConstants::EmptyPrimBatch) override;
    void            CommitLightReservations(LightGroupId) override;
    bool            IsLightCommitted(LightGroupId) const override;
    void            PushLightAttribute(LightGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       MRayInput data) override;
    void            PushLightAttribute(LightGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       std::vector<TextureId>) override;


    CameraGroupId   CreateCameraGroup(std::string typeName) override;
    CameraIdList    ReserveCameras(CameraGroupId, uint32_t count) override;
    void            CommitCamReservations(CameraGroupId) override;
    bool            IsCamCommitted(CameraGroupId) const override;
    void            PushCamAttribute(CameraGroupId, Vector2ui range,
                                     uint32_t attributeIndex,
                                     MRayInput data) override;


    MediumGroupId   CreateMediumGroup(std::string typeName) override;
    MediumIdList    ReserveMediums(MediumGroupId, uint32_t count) override;
    void            CommitMediumReservations(MediumGroupId) override;
    bool            IsMediumCommitted(MediumGroupId) const override;
    void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                                uint32_t attributeIndex,
                                                MRayInput data) override;
    void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                        uint32_t attributeIndex,
                                        std::vector<TextureId> textures) override;


    SurfaceId       CreateSurface(SurfacePrimList primBatches,
                                  SurfaceMatList material,
                                  TransformId = TracerConstants::IdentityTransformId,
                                  OptionalAlphaMapList alphaMaps = TracerConstants::NoAlphaMapList,
                                  CullBackfaceFlagList cullFaceFlags = TracerConstants::CullFaceTrueList) override;
    LightSurfaceId  CreateLightSurface(LightId,
                                       TransformId = TracerConstants::IdentityTransformId,
                                       MediumId = TracerConstants::VacuumMediumId) override;
    CamSurfaceId    CreateCameraSurface(CameraId,
                                        TransformId = TracerConstants::IdentityTransformId,
                                        MediumId = TracerConstants::VacuumMediumId) override;
    void            CommitSurfaces(AcceleratorType) override;


    RendererId  CreateRenderer(std::string typeName) override;
    void        CommitRendererReservations(RendererId) override;
    bool        IsRendererCommitted(RendererId) const override;
    void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                      std::vector<Byte> data) override;

    void        StartRender(RendererId, CamSurfaceId) override;
    void        StoptRender() override;
};

inline TracerMock::TracerMock()
{
}

inline TypeNameList TracerMock::PrimitiveGroups() const
{
    return std::vector<std::string>
    {
        "(P)DefaultTriangle",
        "(P)DefaultTriangleSkinned"
    };
}

inline TypeNameList TracerMock::MaterialGroups() const
{
    return std::vector<std::string>();
}

inline TypeNameList TracerMock::TransformGroups() const
{
    return std::vector<std::string>();
}

inline TypeNameList TracerMock::CameraGroups() const
{
    return std::vector<std::string>();
}

inline TypeNameList TracerMock::MediumGroups() const
{
    return std::vector<std::string>();
}

inline PrimAttributeInfoList TracerMock::AttributeInfo(PrimGroupId) const
{
    return PrimAttributeInfoList{};
}

inline CamAttributeInfoList TracerMock::AttributeInfo(CameraGroupId) const
{
    return CamAttributeInfoList{};
}

inline MediumAttributeInfoList TracerMock::AttributeInfo(MediumGroupId) const
{
    return MediumAttributeInfoList{};
}

inline MatAttributeInfoList TracerMock::AttributeInfo(MatGroupId) const
{
    return MatAttributeInfoList{};
}

inline TransAttributeInfoList TracerMock::AttributeInfo(TransGroupId) const
{
    return TransAttributeInfoList{};
}

inline LightAttributeInfoList TracerMock::AttributeInfo(LightGroupId) const
{
    return LightAttributeInfoList{};
}

inline RendererAttributeInfoList TracerMock::AttributeInfo(RendererId) const
{
    return RendererAttributeInfoList{};
}

inline PrimAttributeInfoList TracerMock::PrimAttributeInfo(std::string_view) const
{
    return PrimAttributeInfoList{};
}

inline CamAttributeInfoList TracerMock::CamAttributeInfo(std::string_view) const
{
    return CamAttributeInfoList{};
}

inline MediumAttributeInfoList TracerMock::MediumAttributeInfo(std::string_view) const
{
    return MediumAttributeInfoList{};
}

inline MatAttributeInfoList TracerMock::MatAttributeInfo(std::string_view) const
{
    return MatAttributeInfoList{};
}

inline TransAttributeInfoList TracerMock::TransAttributeInfo(std::string_view) const
{
    return TransAttributeInfoList{};
}

inline LightAttributeInfoList TracerMock::LightAttributeInfo(std::string_view) const
{
    return LightAttributeInfoList{};
}

inline RendererAttributeInfoList TracerMock::RendererAttributeInfo(std::string_view) const
{
    return RendererAttributeInfoList{};
}

inline std::string TracerMock::TypeName(PrimGroupId) const
{
    return std::string{};
}

inline std::string TracerMock::TypeName(CameraGroupId) const
{
    return std::string{};
}

inline std::string TracerMock::TypeName(MediumGroupId) const
{
    return std::string{};
}

inline std::string TracerMock::TypeName(MatGroupId) const
{
    return std::string{};
}

inline std::string TracerMock::TypeName(TransGroupId) const
{
    return std::string{};
}

inline std::string TracerMock::TypeName(LightGroupId) const
{
    return std::string{};
}

inline std::string TracerMock::TypeName(RendererId) const
{
    return std::string{};
}

inline PrimGroupId TracerMock::CreatePrimitiveGroup(std::string)
{
    return PrimGroupId(0);
}

inline PrimBatchId TracerMock::ReservePrimitiveBatch(PrimGroupId, PrimCount)
{
    return PrimBatchId(0);
}

inline PrimBatchIdList TracerMock::ReservePrimitiveBatches(PrimGroupId,
                                                           std::vector<PrimCount>)
{
    return PrimBatchIdList{};
}

inline void TracerMock::CommitPrimReservations(PrimGroupId)
{
}

inline bool TracerMock::IsPrimCommitted(PrimGroupId) const
{
    return false;
}

inline void TracerMock::PushPrimAttribute(PrimBatchId,
                                          uint32_t,
                                          MRayInput)
{
}

inline void TracerMock::PushPrimAttribute(PrimBatchId,
                                  uint32_t,
                                  Vector2ui,
                                  MRayInput)
{
}

inline MatGroupId TracerMock::CreateMaterialGroup(std::string)
{
    return MatGroupId(0);
}

inline MaterialId TracerMock::ReserveMaterial(MatGroupId,
                                              MediumId,
                                              MediumId)
{
    return MaterialId(0);
}

inline void TracerMock::CommitMatReservations(MatGroupId)
{
}

inline bool TracerMock::IsMatCommitted(MatGroupId) const
{
    return false;
}

inline void TracerMock::PushMatAttribute(MatGroupId, Vector2ui,
                                         uint32_t,
                                         MRayInput)
{
}

inline void TracerMock::PushMatAttribute(MatGroupId, Vector2ui,
                                         uint32_t,
                                         std::vector<TextureId>)
{
}

inline void TracerMock::CommitTexColorSpace(MRayColorSpace)
{
}

inline TextureId TracerMock::CreateTexture2D(Vector2ui, uint32_t,
                                             MRayDataEnum)
{
    return TextureId(0);
}

inline TextureId TracerMock::CreateTexture3D(Vector3ui, uint32_t,
                                MRayDataEnum)
{
    return TextureId(0);
}

inline MRayDataTypeRT TracerMock::GetTexturePixelType(TextureId) const
{
    return MRayDataType<MRayDataEnum::MR_CHAR>{};
}

inline void TracerMock::CommitTextures()
{
}

inline void TracerMock::PushTextureData(TextureId, uint32_t,
                                        MRayInput)
{
}

inline TransGroupId TracerMock::CreateTransformGroup(std::string)
{
    return TransGroupId(0);
}

inline TransformIdList TracerMock::ReserveTransformations(TransGroupId, uint32_t)
{
    return TransformIdList{};
}

inline void TracerMock::CommitTransReservations(TransGroupId)
{
}

inline bool TracerMock::IsTransCommitted(TransGroupId) const
{
    return false;
}

inline void TracerMock::PushTransAttribute(TransGroupId, Vector2ui,
                                           uint32_t,
                                           MRayInput)
{
}

inline LightGroupId TracerMock::CreateLightGroup(std::string,
                                                 PrimGroupId)
{
    return LightGroupId(0);
}

inline LightIdList TracerMock::ReserveLights(LightGroupId,
                                             PrimBatchId)
{
    return LightIdList{};
}

inline void TracerMock::CommitLightReservations(LightGroupId)
{
}

inline bool TracerMock::IsLightCommitted(LightGroupId) const
{
    return false;
}

inline void TracerMock::PushLightAttribute(LightGroupId, Vector2ui,
                                           uint32_t,
                                           MRayInput)
{
}

inline void TracerMock::PushLightAttribute(LightGroupId, Vector2ui,
                                   uint32_t,
                                   std::vector<TextureId>)
{
}

inline CameraGroupId TracerMock::CreateCameraGroup(std::string)
{
    return CameraGroupId(0);
}

inline CameraIdList TracerMock::ReserveCameras(CameraGroupId, uint32_t)
{
    return CameraIdList{};
}

inline void TracerMock::CommitCamReservations(CameraGroupId)
{
}

inline bool TracerMock::IsCamCommitted(CameraGroupId) const
{
    return false;
}

inline void TracerMock::PushCamAttribute(CameraGroupId, Vector2ui,
                                         uint32_t,
                                         MRayInput)
{

}

inline MediumGroupId TracerMock::CreateMediumGroup(std::string)
{
    return MediumGroupId(0);
}

inline MediumIdList TracerMock::ReserveMediums(MediumGroupId, uint32_t)
{
    return MediumIdList{};
}

inline void TracerMock::CommitMediumReservations(MediumGroupId)
{
}

inline bool TracerMock::IsMediumCommitted(MediumGroupId) const
{
    return false;
}

inline void TracerMock::PushMediumAttribute(MediumGroupId, Vector2ui,
                                    uint32_t,
                                    MRayInput)
{

}

inline void TracerMock::PushMediumAttribute(MediumGroupId, Vector2ui,
                                            uint32_t,
                                            std::vector<TextureId>)
{

}

inline SurfaceId TracerMock::CreateSurface(SurfacePrimList,
                                           SurfaceMatList,
                                           TransformId,
                                           OptionalAlphaMapList,
                                           CullBackfaceFlagList)
{
    return SurfaceId{};
}

inline LightSurfaceId TracerMock::CreateLightSurface(LightId,
                                                     TransformId,
                                                     MediumId)
{
    return LightSurfaceId{0};
}

inline CamSurfaceId TracerMock::CreateCameraSurface(CameraId,
                                             TransformId,
                                             MediumId)
{
    return CamSurfaceId{0};
}

inline void TracerMock::CommitSurfaces(AcceleratorType)
{
}

inline RendererId TracerMock::CreateRenderer(std::string typeName)
{
    return RendererId{};
}

inline void TracerMock::CommitRendererReservations(RendererId)
{

}

inline bool TracerMock::IsRendererCommitted(RendererId) const
{
    return false;
}

inline void TracerMock::PushRendererAttribute(RendererId, uint32_t,
                                              std::vector<Byte>)
{

}

inline void TracerMock::StartRender(RendererId, CamSurfaceId)
{
}

inline void TracerMock::StoptRender()
{
}