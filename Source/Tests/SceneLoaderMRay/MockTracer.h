#pragma once

#include "Core/TracerI.h"
#include "TransientPool/TransientPool.h"

#include <map>

// Mock tracer has
struct PrimMockPack
{
    PrimAttributeInfoList attribInfo;
};

struct CamMockPack
{
    CamAttributeInfoList attribInfo;
};

struct MedMockPack
{
    MediumAttributeInfoList attribInfo;
};

struct MatMockPack
{
    MatAttributeInfoList attribInfo;
};

struct TransMockPack
{
    TransAttributeInfoList attribInfo;
};

struct LightMockPack
{
    LightAttributeInfoList attribInfo;
};

class TracerMock : public TracerI
{
    private:
    std::map<PrimGroupId, const PrimAttributeInfoList&>     primAttribMap;
    std::map<CameraGroupId, const CamAttributeInfoList&>    camAttribMap;
    std::map<MediumGroupId, const MediumAttributeInfoList&> mediumAttribMap;
    std::map<MatGroupId, const MatAttributeInfoList&>       matAttribMap;
    std::map<TransGroupId, const TransAttributeInfoList&>   transAttribMap;
    std::map<LightGroupId, const LightAttributeInfoList&>   lightAttribMap;

    // Mock packs
    std::map<std::string_view, PrimMockPack>    primMockPack;
    std::map<std::string_view, CamMockPack>     camMockPack;
    std::map<std::string_view, MedMockPack>     medMockPack;
    std::map<std::string_view, MatMockPack>     matMockPack;
    std::map<std::string_view, TransMockPack>   transMockPack;
    std::map<std::string_view, LightMockPack>   lightMockPack;

    bool                unqiueGroups;

    public:
                        TracerMock(bool uniqueGroups = true);

    TypeNameList        PrimitiveGroups() const override;
    TypeNameList        MaterialGroups() const override;
    TypeNameList        TransformGroups() const override;
    TypeNameList        CameraGroups() const override;
    TypeNameList        MediumGroups() const override;
    TypeNameList        LightGroups() const override;

    PrimAttributeInfoList       AttributeInfo(PrimGroupId) const override;
    CamAttributeInfoList        AttributeInfo(CameraGroupId) const override;
    MediumAttributeInfoList     AttributeInfo(MediumGroupId) const override;
    MatAttributeInfoList        AttributeInfo(MatGroupId) const override;
    TransAttributeInfoList      AttributeInfo(TransGroupId) const override;
    LightAttributeInfoList      AttributeInfo(LightGroupId) const override;
    RendererAttributeInfoList   AttributeInfo(RendererId) const override;

    PrimAttributeInfoList       AttributeInfoPrim(std::string_view) const override;
    CamAttributeInfoList        AttributeInfoCam(std::string_view) const override;
    MediumAttributeInfoList     AttributeInfoMedium(std::string_view) const override;
    MatAttributeInfoList        AttributeInfoMat(std::string_view) const override;
    TransAttributeInfoList      AttributeInfoTrans(std::string_view) const override;
    LightAttributeInfoList      AttributeInfoLight(std::string_view) const override;
    RendererAttributeInfoList   AttributeInfoRenderer(std::string_view) const override;

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
                                      TransientData data) override;
    void            PushPrimAttribute(PrimBatchId,
                                      uint32_t attributeIndex,
                                      Vector2ui subBatchRange,
                                      TransientData data) override;

    MatGroupId      CreateMaterialGroup(std::string typeName) override;
    MaterialId      ReserveMaterial(MatGroupId, AttributeCountList,
                                    MediumPair = TracerConstants::VacuumMediumPair) override;
    MaterialIdList  ReserveMaterials(MatGroupId,
                                     std::vector<AttributeCountList>,
                                     std::vector<MediumPair> = {}) override;

    void            CommitMatReservations(MatGroupId) override;
    bool            IsMatCommitted(MatGroupId) const override;
    void            PushMatAttribute(MatGroupId, Vector2ui range,
                                     uint32_t attributeIndex,
                                     TransientData data) override;
    void            PushMatAttribute(MatGroupId, Vector2ui range,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>> textures) override;
    void            PushMatAttribute(MatGroupId, Vector2ui range,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>) override;

    void            CommitTexColorSpace(MRayColorSpaceEnum = MRayColorSpaceEnum::MR_DEFAULT) override;
    TextureId       CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                    MRayPixelEnum pixelType) override;
    TextureId       CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                    MRayPixelEnum pixelType) override;
    MRayDataTypeRT  GetTexturePixelType(TextureId) const override;
    void            CommitTextures() override;
    void            PushTextureData(TextureId, uint32_t mipLevel,
                                    TransientData data) override;


    TransGroupId    CreateTransformGroup(std::string typeName) override;
    TransformId     ReserveTransformation(TransGroupId, AttributeCountList) override;
    TransformIdList ReserveTransformations(TransGroupId, std::vector<AttributeCountList>) override;
    void            CommitTransReservations(TransGroupId) override;
    bool            IsTransCommitted(TransGroupId) const override;
    void            PushTransAttribute(TransGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       TransientData data) override;


    LightGroupId    CreateLightGroup(std::string typeName) override;
    LightId         ReserveLight(LightGroupId, AttributeCountList,
                                 PrimBatchId = TracerConstants::EmptyPrimBatch) override;
    LightIdList     ReserveLights(LightGroupId, std::vector<AttributeCountList>,
                                  std::vector<PrimBatchId> = std::vector<PrimBatchId>{}) override;
    void            CommitLightReservations(LightGroupId) override;
    bool            IsLightCommitted(LightGroupId) const override;
    void            PushLightAttribute(LightGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       TransientData data) override;
    void            PushLightAttribute(LightGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       TransientData,
                                       std::vector<Optional<TextureId>> textures) override;
    void            PushLightAttribute(LightGroupId, Vector2ui range,
                                       uint32_t attributeIndex,
                                       std::vector<TextureId>) override;


    CameraGroupId   CreateCameraGroup(std::string typeName) override;
    CameraId        ReserveCamera(CameraGroupId, AttributeCountList) override;
    CameraIdList    ReserveCameras(CameraGroupId, std::vector<AttributeCountList>) override;
    void            CommitCamReservations(CameraGroupId) override;
    bool            IsCamCommitted(CameraGroupId) const override;
    void            PushCamAttribute(CameraGroupId, Vector2ui range,
                                     uint32_t attributeIndex,
                                     TransientData data) override;


    MediumGroupId   CreateMediumGroup(std::string typeName) override;
    MediumId        ReserveMedium(MediumGroupId, AttributeCountList) override;
    MediumIdList    ReserveMediums(MediumGroupId, std::vector<AttributeCountList>) override;
    void            CommitMediumReservations(MediumGroupId) override;
    bool            IsMediumCommitted(MediumGroupId) const override;
    void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                        uint32_t attributeIndex,
                                        TransientData data) override;
    void            PushMediumAttribute(MediumGroupId, Vector2ui range,
                                        uint32_t attributeIndex,
                                        TransientData,
                                        std::vector<Optional<TextureId>> textures) override;
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
                                      TransientData data) override;

    void        StartRender(RendererId, CamSurfaceId) override;
    void        StoptRender() override;
};

inline TracerMock::TracerMock(bool ug)
    : unqiueGroups(ug)
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    using enum AttributeTexturable;

    // =================== //
    //     Primitives      //
    // =================== //
    primMockPack["(P)DefaultTriangle"] = PrimMockPack
    {
        .attribInfo = PrimAttributeInfoList
        {
            PrimAttributeInfo(POSITION, MRayDataType<MR_VECTOR_3>(),    IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(NORMAL,   MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(UV0,      MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(INDEX,    MRayDataType<MR_VECTOR_3UI>(),  IS_SCALAR, MR_MANDATORY)
        }
    };

    primMockPack["(P)DefaultTriangleSkinned"] = PrimMockPack
    {
        .attribInfo = PrimAttributeInfoList
        {
            PrimAttributeInfo(POSITION,     MRayDataType<MR_VECTOR_3>(),    IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(NORMAL,       MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(UV0,          MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(WEIGHT,       MRayDataType<MR_UNORM_4x8>(),   IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(WEIGHT_INDEX, MRayDataType<MR_VECTOR_4UC>(),  IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(INDEX,        MRayDataType<MR_VECTOR_3UI>(),  IS_SCALAR, MR_MANDATORY)
        }
    };

    // =================== //
    //     Materials       //
    // =================== //
    matMockPack["(Mt)Lambert"] = MatMockPack
    {
        .attribInfo = MatAttributeInfoList
        {
            MatAttributeInfo("albedo", MRayDataType<MR_VECTOR_3>(),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
            MatAttributeInfo("normalMap", MRayDataType<MR_VECTOR_3>(),
                             IS_SCALAR, MR_OPTIONAL, MR_TEXTURE_ONLY)
        }
    };
    matMockPack["(Mt)Unreal"] = MatMockPack
    {
        .attribInfo = MatAttributeInfoList
        {
            MatAttributeInfo("albedo", MRayDataType<MR_VECTOR_3>(),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
            MatAttributeInfo("metallic", MRayDataType<MR_DEFAULT_FLT>(),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
            MatAttributeInfo("specular", MRayDataType<MR_DEFAULT_FLT>(),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
            MatAttributeInfo("roughness", MRayDataType<MR_DEFAULT_FLT>(),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
            MatAttributeInfo("normalMap", MRayDataType<MR_VECTOR_3>(),
                             IS_SCALAR, MR_OPTIONAL, MR_TEXTURE_ONLY)
        }
    };

    // =================== //
    //       Lights        //
    // =================== //
    lightMockPack["(L)Null"] = LightMockPack{};
    lightMockPack["(L)Skysphere"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT)
        }
    };
    lightMockPack["(L)Primitive(P)Triangle"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT)
        }
    };
    lightMockPack["(L)Rectangle"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
            LightAttributeInfo("position", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_CONSTANT_ONLY),
            LightAttributeInfo("right", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_CONSTANT_ONLY),
            LightAttributeInfo("up", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_CONSTANT_ONLY)
        }
    };

    // =================== //
    //       Cameras       //
    // =================== //
    camMockPack["(C)Pinhole"] = CamMockPack
    {
        .attribInfo = CamAttributeInfoList
        {
            CamAttributeInfo("fov", MRayDataType<MR_DEFAULT_FLT>(), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("isFovX", MRayDataType<MR_CHAR>(), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("planes", MRayDataType<MR_VECTOR_2>(), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("gaze", MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("position", MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("up", MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY)
        }
    };

    // =================== //
    //     Transforms      //
    // =================== //
    transMockPack["(T)Identity"] = TransMockPack {};
    transMockPack["(T)Single"] = TransMockPack
    {
        .attribInfo = TransAttributeInfoList
        {
            TransAttributeInfo("matrix", MRayDataType<MR_MATRIX_4x4>(), IS_SCALAR, MR_MANDATORY)
        }
    };
    transMockPack["(T)Multi"] = TransMockPack
    {
        .attribInfo = TransAttributeInfoList
        {
            TransAttributeInfo("matrix", MRayDataType<MR_MATRIX_4x4>(), IS_ARRAY, MR_MANDATORY)
        }
    };

    // =================== //
    //       Mediums       //
    // =================== //
    medMockPack["(Md)Vacuum"] = MedMockPack{};
    medMockPack["(Md)Homogeneous"] = MedMockPack
    {
        .attribInfo = TexturedAttributeInfoList
        {
            MediumAttributeInfo("absorbtion", MRayDataType<MR_VECTOR_3>(),
                                IS_SCALAR, MR_MANDATORY, MR_CONSTANT_ONLY),
            MediumAttributeInfo("ior", MRayDataType<MR_DEFAULT_FLT>(),
                                IS_SCALAR, MR_MANDATORY, MR_CONSTANT_ONLY)
        }
    };
}

inline TypeNameList TracerMock::PrimitiveGroups() const
{
    return std::vector<std::string>
    {
        "(P)Triangle",
        "(P)TriangleSkinned"
    };
}

inline TypeNameList TracerMock::MaterialGroups() const
{
    return std::vector<std::string>
    {
        "(Mt)Lambert",
        "(Mt)Unreal"
    };
}

inline TypeNameList TracerMock::TransformGroups() const
{
    return std::vector<std::string>
    {
        "(T)Identity",
        "(T)Single",
        "(T)Multi"
    };
}

inline TypeNameList TracerMock::CameraGroups() const
{
    return std::vector<std::string>
    {
        "(C)Pinhole"
    };
}

inline TypeNameList TracerMock::MediumGroups() const
{
    return std::vector<std::string>
    {
        "(Md)Vacuum",
        "(Md)Homogeneous"
    };
}

inline TypeNameList TracerMock::LightGroups() const
{
    return std::vector<std::string>
    {
        "(L)Null",
        "(L)Skysphere",
        "(L)Rectangle",
        "(L)Primitive(P)Triangle"
    };
}

inline PrimAttributeInfoList TracerMock::AttributeInfo(PrimGroupId id) const
{
    return primAttribMap.at(id);
}

inline CamAttributeInfoList TracerMock::AttributeInfo(CameraGroupId id) const
{
    return camAttribMap.at(id);
}

inline MediumAttributeInfoList TracerMock::AttributeInfo(MediumGroupId id) const
{
    return mediumAttribMap.at(id);
}

inline MatAttributeInfoList TracerMock::AttributeInfo(MatGroupId id) const
{
    return matAttribMap.at(id);
}

inline TransAttributeInfoList TracerMock::AttributeInfo(TransGroupId id) const
{
    return transAttribMap.at(id);
}

inline LightAttributeInfoList TracerMock::AttributeInfo(LightGroupId id) const
{
    return lightAttribMap.at(id);
}

inline RendererAttributeInfoList TracerMock::AttributeInfo(RendererId) const
{
    return RendererAttributeInfoList{};
}

inline PrimAttributeInfoList TracerMock::AttributeInfoPrim(std::string_view) const
{
    return PrimAttributeInfoList{};
}

inline CamAttributeInfoList TracerMock::AttributeInfoCam(std::string_view) const
{
    return CamAttributeInfoList{};
}

inline MediumAttributeInfoList TracerMock::AttributeInfoMedium(std::string_view) const
{
    return MediumAttributeInfoList{};
}

inline MatAttributeInfoList TracerMock::AttributeInfoMat(std::string_view) const
{
    return MatAttributeInfoList{};
}

inline TransAttributeInfoList TracerMock::AttributeInfoTrans(std::string_view) const
{
    return TransAttributeInfoList{};
}

inline LightAttributeInfoList TracerMock::AttributeInfoLight(std::string_view) const
{
    return LightAttributeInfoList{};
}

inline RendererAttributeInfoList TracerMock::AttributeInfoRenderer(std::string_view) const
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

inline PrimBatchIdList TracerMock::ReservePrimitiveBatches(PrimGroupId, std::vector<PrimCount>)
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
                                          TransientData)
{
}

inline void TracerMock::PushPrimAttribute(PrimBatchId,
                                          uint32_t,
                                          Vector2ui,
                                          TransientData)
{
}

inline MatGroupId TracerMock::CreateMaterialGroup(std::string)
{
    return MatGroupId(0);
}

inline MaterialId TracerMock::ReserveMaterial(MatGroupId, AttributeCountList,
                                              MediumPair)
{
    return MaterialId(0);
}

inline MaterialIdList TracerMock::ReserveMaterials(MatGroupId,
                                                   std::vector<AttributeCountList>,
                                                   std::vector<MediumPair>)
{
    return MaterialIdList{};
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
                                         TransientData)
{
}

inline void TracerMock::PushMatAttribute(MatGroupId, Vector2ui,
                                         uint32_t,
                                         TransientData,
                                         std::vector<Optional<TextureId>>)
{

}

inline void TracerMock::PushMatAttribute(MatGroupId, Vector2ui,
                                         uint32_t,
                                         std::vector<TextureId>)
{
}

inline void TracerMock::CommitTexColorSpace(MRayColorSpaceEnum)
{
}

inline TextureId TracerMock::CreateTexture2D(Vector2ui, uint32_t,
                                             MRayPixelEnum)
{
    return TextureId{0};
}

inline TextureId TracerMock::CreateTexture3D(Vector3ui, uint32_t,
                                             MRayPixelEnum)
{
    return TextureId{0};
}

inline MRayDataTypeRT TracerMock::GetTexturePixelType(TextureId) const
{
    return MRayDataType<MRayDataEnum::MR_CHAR>{};
}

inline void TracerMock::CommitTextures()
{
}

inline void TracerMock::PushTextureData(TextureId, uint32_t,
                                        TransientData)
{
}

inline TransGroupId TracerMock::CreateTransformGroup(std::string)
{
    return TransGroupId(0);
}

inline TransformId TracerMock::ReserveTransformation(TransGroupId, AttributeCountList)
{
    return TransformId(0);
}

inline TransformIdList TracerMock::ReserveTransformations(TransGroupId, std::vector<AttributeCountList>)
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
                                           TransientData)
{
}

inline LightGroupId TracerMock::CreateLightGroup(std::string)
{
    return LightGroupId(0);
}

inline LightId TracerMock::ReserveLight(LightGroupId,
                                        AttributeCountList,
                                        PrimBatchId)
{
    return LightId(0);
}

inline LightIdList TracerMock::ReserveLights(LightGroupId,
                                             std::vector<AttributeCountList>,
                                             std::vector<PrimBatchId>)
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
                                           TransientData)
{
}

inline void TracerMock::PushLightAttribute(LightGroupId, Vector2ui,
                                    uint32_t,
                                    TransientData,
                                    std::vector<Optional<TextureId>>)
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

inline CameraId TracerMock::ReserveCamera(CameraGroupId,
                                          AttributeCountList)
{
    return CameraId(0);
}

inline CameraIdList TracerMock::ReserveCameras(CameraGroupId,
                                               std::vector<AttributeCountList>)
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
                                         TransientData)
{

}

inline MediumGroupId TracerMock::CreateMediumGroup(std::string)
{
    return MediumGroupId(0);
}

inline MediumId TracerMock::ReserveMedium(MediumGroupId, AttributeCountList)
{
    return MediumId(0);
}

inline MediumIdList TracerMock::ReserveMediums(MediumGroupId,
                                               std::vector<AttributeCountList>)
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
                                            TransientData)
{

}

inline void TracerMock::PushMediumAttribute(MediumGroupId, Vector2ui,
                                            uint32_t,
                                            TransientData,
                                            std::vector<Optional<TextureId>>)
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
                                              TransientData)
{

}

inline void TracerMock::StartRender(RendererId, CamSurfaceId)
{

}

inline void TracerMock::StoptRender()
{
}