#pragma once

#include "Core/TracerI.h"
#include "TransientPool/TransientPool.h"

#include <map>

size_t AcquireTransientDataCount(MRayDataTypeRT dataTypeRT,
                                 const TransientData& data)
{
    return std::visit([&data](auto&& type)
    {
        using T = typename std::remove_cvref_t<decltype(type)>::Type;
        return data.AccessAs<const T>().size();
    }, dataTypeRT);
}



// Mock tracer has
struct PrimMockPack
{
    PrimAttributeInfoList   attribInfo;
    std::string_view        name;
};

struct CamMockPack
{
    CamAttributeInfoList    attribInfo;
    std::string_view        name;
};

struct MedMockPack
{
    MediumAttributeInfoList attribInfo;
    std::string_view        name;
};

struct MatMockPack
{
    MatAttributeInfoList    attribInfo;
    std::string_view        name;
};

struct TransMockPack
{
    TransAttributeInfoList  attribInfo;
    std::string_view        name;
};

struct LightMockPack
{
    LightAttributeInfoList  attribInfo;
    std::string_view        name;
};

struct PrimGroupMock
{
    const PrimMockPack&         mp;
    PrimGroupId                 id;
    std::vector<PrimBatchId>    batchList;
    std::atomic_size_t          batchCounter;
    bool                        isComitted;
};

struct MatGroupMock
{
    const MatMockPack&      mp;
    MatGroupId              id;
    std::vector<MaterialId> matList;
    std::atomic_size_t      idCounter;
    bool                    isComitted;
};

struct MediumGroupMock
{
    const MedMockPack&      mp;
    MediumGroupId           id;
    std::vector<MediumId>   medList;
    std::atomic_size_t      idCounter;
    bool                    isComitted;
};

struct TransGroupMock
{
    const TransMockPack&        mp;
    MediumGroupId               id;
    std::vector<TransformId>    transList;
    std::atomic_size_t          idCounter;
    bool                        isComitted;
};

struct LightGroupMock
{
    const LightMockPack&    mp;
    LightGroupId            id;
    std::vector<LightId>    lightList;
    std::atomic_size_t      idCounter;
    bool                    isComitted;
};

struct CamGroupMock
{
    const CamMockPack&      mp;
    LightGroupId            id;
    std::vector<CameraId>   camList;
    std::atomic_size_t      idCounter;
    bool                    isComitted;
};

class TracerMock : public TracerI
{
    private:
    std::map<PrimGroupId, PrimGroupMock>        primGroups;
    std::map<CameraGroupId, CamGroupMock>       camGroups;
    std::map<MediumGroupId, MediumGroupMock>    mediumGroups;
    std::map<MatGroupId, MatGroupMock>          matGroups;
    std::map<TransGroupId, TransGroupMock>      transGroups;
    std::map<LightGroupId, LightGroupMock>      lightGroups;

    // Mock packs
    std::map<std::string_view, PrimMockPack>    primMockPack;
    std::map<std::string_view, CamMockPack>     camMockPack;
    std::map<std::string_view, MedMockPack>     medMockPack;
    std::map<std::string_view, MatMockPack>     matMockPack;
    std::map<std::string_view, TransMockPack>   transMockPack;
    std::map<std::string_view, LightMockPack>   lightMockPack;

    size_t primGroupCounter;
    size_t camGroupCounter;
    size_t mediumGroupCounter;
    size_t matGroupCounter;
    size_t transGroupCounter;
    size_t lightGroupCounter;

    mutable std::mutex pGLock;
    mutable std::mutex cGLock;
    mutable std::mutex meGLock;
    mutable std::mutex mtGLock;
    mutable std::mutex tGLock;
    mutable std::mutex lGLock;

    // Properties
    bool                print;

    public:
                        TracerMock(bool print= true);

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
    void            PushPrimAttribute(PrimGroupId, PrimBatchId,
                                      uint32_t attributeIndex,
                                      TransientData data) override;
    void            PushPrimAttribute(PrimGroupId, PrimBatchId,
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

inline TracerMock::TracerMock(bool pl)
    : print(pl)
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    using enum AttributeTexturable;

    // =================== //
    //     Primitives      //
    // =================== //
    primMockPack["(P)Triangle"] = PrimMockPack
    {
        .attribInfo = PrimAttributeInfoList
        {
            PrimAttributeInfo(POSITION, MRayDataType<MR_VECTOR_3>(),    IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(NORMAL,   MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(UV0,      MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(INDEX,    MRayDataType<MR_VECTOR_3UI>(),  IS_SCALAR, MR_MANDATORY)
        },
        .name = "(P)Triangle"
    };

    primMockPack["(P)TriangleSkinned"] = PrimMockPack
    {
        .attribInfo = PrimAttributeInfoList
        {
            PrimAttributeInfo(POSITION,     MRayDataType<MR_VECTOR_3>(),    IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(NORMAL,       MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(UV0,          MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(WEIGHT,       MRayDataType<MR_UNORM_4x8>(),   IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(WEIGHT_INDEX, MRayDataType<MR_VECTOR_4UC>(),  IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(INDEX,        MRayDataType<MR_VECTOR_3UI>(),  IS_SCALAR, MR_MANDATORY)
        },
        .name = "(P)TriangleSkinned"
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
        },
        .name = "(Mt)Lambert"
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
        },
        .name = "(Mt)Unreal"
    };

    // =================== //
    //       Lights        //
    // =================== //
    lightMockPack["(L)Null"] = LightMockPack
    {
        .attribInfo = {},
        .name = "(L)Null"
    };
    lightMockPack["(L)Skysphere"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT)
        },
        .name = "(L)Skysphere"
    };
    lightMockPack["(L)Primitive(P)Triangle"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT)
        },
        .name ="(L)Primitive(P)Triangle"
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
        },
        .name = "(L)Rectangle"
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
        },
        .name = "(C)Pinhole"
    };

    // =================== //
    //     Transforms      //
    // =================== //
    transMockPack["(T)Identity"] = TransMockPack
    {
        .attribInfo = {},
        .name = "(T)Identity"
    };
    transMockPack["(T)Single"] = TransMockPack
    {
        .attribInfo = TransAttributeInfoList
        {
            TransAttributeInfo("matrix", MRayDataType<MR_MATRIX_4x4>(), IS_SCALAR, MR_MANDATORY)
        },
        .name = "(T)Single"
    };
    transMockPack["(T)Multi"] = TransMockPack
    {
        .attribInfo = TransAttributeInfoList
        {
            TransAttributeInfo("matrix", MRayDataType<MR_MATRIX_4x4>(), IS_ARRAY, MR_MANDATORY)
        },
        .name = "(T)Multi"
    };

    // =================== //
    //       Mediums       //
    // =================== //
    medMockPack["(Md)Vacuum"] = MedMockPack
    {
        .attribInfo = {},
        .name = "(Md)Vacuum"
    };
    medMockPack["(Md)Homogeneous"] = MedMockPack
    {
        .attribInfo = TexturedAttributeInfoList
        {
            MediumAttributeInfo("absorbtion", MRayDataType<MR_VECTOR_3>(),
                                IS_SCALAR, MR_MANDATORY, MR_CONSTANT_ONLY),
            MediumAttributeInfo("ior", MRayDataType<MR_DEFAULT_FLT>(),
                                IS_SCALAR, MR_MANDATORY, MR_CONSTANT_ONLY)
        },
        .name = "(Md)Homogeneous"
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
    return primGroups.at(id).mp.attribInfo;
}

inline CamAttributeInfoList TracerMock::AttributeInfo(CameraGroupId id) const
{
    return camGroups.at(id).mp.attribInfo;
}

inline MediumAttributeInfoList TracerMock::AttributeInfo(MediumGroupId id) const
{
    return mediumGroups.at(id).mp.attribInfo;
}

inline MatAttributeInfoList TracerMock::AttributeInfo(MatGroupId id) const
{
    return matGroups.at(id).mp.attribInfo;
}

inline TransAttributeInfoList TracerMock::AttributeInfo(TransGroupId id) const
{
    return transGroups.at(id).mp.attribInfo;
}

inline LightAttributeInfoList TracerMock::AttributeInfo(LightGroupId id) const
{
    return lightGroups.at(id).mp.attribInfo;
}

inline RendererAttributeInfoList TracerMock::AttributeInfo(RendererId) const
{
    return RendererAttributeInfoList{};
}

inline PrimAttributeInfoList TracerMock::AttributeInfoPrim(std::string_view name) const
{
    return primMockPack.at(name).attribInfo;
}

inline CamAttributeInfoList TracerMock::AttributeInfoCam(std::string_view name) const
{
    return camMockPack.at(name).attribInfo;
}

inline MediumAttributeInfoList TracerMock::AttributeInfoMedium(std::string_view name) const
{
    return medMockPack.at(name).attribInfo;
}

inline MatAttributeInfoList TracerMock::AttributeInfoMat(std::string_view name) const
{
    return matMockPack.at(name).attribInfo;
}

inline TransAttributeInfoList TracerMock::AttributeInfoTrans(std::string_view name) const
{
    return transMockPack.at(name).attribInfo;
}

inline LightAttributeInfoList TracerMock::AttributeInfoLight(std::string_view name) const
{
    return lightMockPack.at(name).attribInfo;
}

inline RendererAttributeInfoList TracerMock::AttributeInfoRenderer(std::string_view) const
{
    return RendererAttributeInfoList{};
}

inline std::string TracerMock::TypeName(PrimGroupId id) const
{
    return std::string(primGroups.at(id).mp.name);
}

inline std::string TracerMock::TypeName(CameraGroupId id) const
{
    return std::string(camGroups.at(id).mp.name);
}

inline std::string TracerMock::TypeName(MediumGroupId id) const
{
    return std::string(mediumGroups.at(id).mp.name);
}

inline std::string TracerMock::TypeName(MatGroupId id) const
{
    return std::string(matGroups.at(id).mp.name);
}

inline std::string TracerMock::TypeName(TransGroupId id) const
{
    return std::string(transGroups.at(id).mp.name);
}

inline std::string TracerMock::TypeName(LightGroupId id) const
{
    return std::string(lightGroups.at(id).mp.name);
}

inline std::string TracerMock::TypeName(RendererId) const
{
    return std::string{};
}

inline PrimGroupId TracerMock::CreatePrimitiveGroup(std::string name)
{
    auto loc = primMockPack.find(name);
    if(loc == primMockPack.cend())
        throw MRayError("Failed to find primitive type: {}", name);


    std::lock_guard<std::mutex> lock(pGLock);
    PrimGroupId id = static_cast<PrimGroupId>(primGroupCounter++);
    primGroups.try_emplace(id, loc->second, id,
                           std::vector<PrimBatchId>(),
                           0, false);
    return id;
}

inline PrimBatchId TracerMock::ReservePrimitiveBatch(PrimGroupId id, PrimCount count)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(pGLock);
        atomicCounter = &primGroups.at(id).batchCounter;
    }

    if(print)
        MRAY_LOG("Reserving primitive over PrimGroup({}), VCount: {}, PCount: {}",
                 static_cast<uint32_t>(id), count.attributeCount, count.primCount);

    size_t batchId = atomicCounter->fetch_add(1);
    return PrimBatchId(batchId);
}

inline PrimBatchIdList TracerMock::ReservePrimitiveBatches(PrimGroupId, std::vector<PrimCount>)
{
    return PrimBatchIdList{};
}

inline void TracerMock::CommitPrimReservations(PrimGroupId id)
{
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(pGLock);
        isCommitted = &primGroups.at(id).isComitted;
    }

    if(*isCommitted)
        throw MRayError("PrimitiveGroup({}) is already comitted!",
                        static_cast<uint32_t>(id));
    else
        *isCommitted = true;
}

inline bool TracerMock::IsPrimCommitted(PrimGroupId id) const
{
    std::lock_guard<std::mutex> lock(pGLock);
    return primGroups.at(id).isComitted;
}

inline void TracerMock::PushPrimAttribute(PrimGroupId gId,
                                          PrimBatchId batchId,
                                          uint32_t attribIndex,
                                          TransientData data)
{
    if(!print) return;
    std::lock_guard<std::mutex> lock(pGLock);

    const auto& attribType = primGroups.at(gId).mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = std::get<PrimAttributeInfo::LAYOUT_INDEX>(attribType);
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);
    MRAY_LOG("Pushing prim attribute of ({}:{}),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<uint32_t>(gId), static_cast<uint32_t>(batchId),
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushPrimAttribute(PrimGroupId gId,
                                          PrimBatchId batchId,
                                          uint32_t attribIndex,
                                          Vector2ui subBatchRange,
                                          TransientData data)
{
    if(!print) return;
    std::lock_guard<std::mutex> lock(pGLock);

    const auto& attribType = primGroups.at(gId).mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = std::get<PrimAttributeInfo::LAYOUT_INDEX>(attribType);
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);
    MRAY_LOG("Pushing prim attribute of ({}:{})->[{}, {}],"
             "DataType: {:s}, ByteSize: {}",
             static_cast<uint32_t>(gId), static_cast<uint32_t>(batchId),
             subBatchRange[0], subBatchRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline MatGroupId TracerMock::CreateMaterialGroup(std::string)
{
    //auto loc = primMockPack.find(name);
    //if(loc == primMockPack.cend())
    //    throw MRayError("Failed to find primitive type: {}", name);
    //return static_cast<PrimGroupId>(primGroupCounter.fetch_add(1));

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