#pragma once

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "TransientPool/TransientPool.h"

#include <map>

static size_t AcquireTransientDataCount(MRayDataTypeRT dataTypeRT,
                                        const TransientData& data)
{
    assert(data.ByteSize() % dataTypeRT.Size() == 0);
    return data.ByteSize() / dataTypeRT.Size();

    // return dataTypeRT.SwitchCase([&data](auto&& type)
    // {
    //     using T = typename std::remove_cvref_t<decltype(type)>::Type;
    //     return data.AccessAs<const T>().size();
    // });
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
    TransGroupId                id;
    std::vector<TransformId>    transList;
    std::atomic_size_t          idCounter;
    bool                        isComitted;
};

struct LightGroupMock
{
    const LightMockPack&    mp;
    LightGroupId            id;
    PrimGroupId             primGId;
    std::vector<LightId>    lightList;
    std::atomic_size_t      idCounter;
    bool                    isComitted;
};

struct CamGroupMock
{
    const CamMockPack&      mp;
    CameraGroupId           id;
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

    size_t primGroupCounter     = 0;
    size_t camGroupCounter      = 0;
    size_t mediumGroupCounter   = 0;
    size_t matGroupCounter      = 0;
    size_t transGroupCounter    = 0;
    size_t lightGroupCounter    = 0;

    // Surface Related
    std::atomic_size_t  surfaceCounter      = 0;
    std::atomic_size_t  lightSurfaceCounter = 0;
    std::atomic_size_t  camSurfaceCounter   = 0;
    std::atomic_size_t  volumeCounter       = 0;

    // Texture Related
    std::atomic_size_t  textureCounter  = 0;
    bool                globalTexCommit = false;

    mutable std::mutex pGLock;
    mutable std::mutex cGLock;
    mutable std::mutex meGLock;
    mutable std::mutex mtGLock;
    mutable std::mutex tGLock;
    mutable std::mutex lGLock;

    TracerParameters params = {};

    // Properties
    bool                print;

    public:
                        TracerMock(bool print = true);

    TypeNameList        PrimitiveGroups() const override;
    TypeNameList        MaterialGroups() const override;
    TypeNameList        TransformGroups() const override;
    TypeNameList        CameraGroups() const override;
    TypeNameList        MediumGroups() const override;
    TypeNameList        LightGroups() const override;
    TypeNameList        Renderers() const override;

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
    void            TransformPrimitives(PrimGroupId gId,
                                        std::vector<PrimBatchId> primBatchIds,
                                        std::vector<Matrix3x4> transforms) override;

    MatGroupId      CreateMaterialGroup(std::string typeName) override;
    MaterialId      ReserveMaterial(MatGroupId, AttributeCountList) override;
    MaterialIdList  ReserveMaterials(MatGroupId, std::vector<AttributeCountList>) override;

    void            CommitMatReservations(MatGroupId) override;
    bool            IsMatCommitted(MatGroupId) const override;
    void            PushMatAttribute(MatGroupId, CommonIdRange range,
                                     uint32_t attributeIndex,
                                     TransientData data) override;
    void            PushMatAttribute(MatGroupId, CommonIdRange range,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>> textures) override;
    void            PushMatAttribute(MatGroupId, CommonIdRange range,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>) override;

    TextureId       CreateTexture2D(Vector2ui size, uint32_t mipCount,
                                    MRayTextureParameters) override;
    TextureId       CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                    MRayTextureParameters) override;
    void            CommitTextures() override;
    void            PushTextureData(TextureId, uint32_t mipLevel,
                                    TransientData data) override;


    TransGroupId    CreateTransformGroup(std::string typeName) override;
    TransformId     ReserveTransformation(TransGroupId, AttributeCountList) override;
    TransformIdList ReserveTransformations(TransGroupId, std::vector<AttributeCountList>) override;
    void            CommitTransReservations(TransGroupId) override;
    bool            IsTransCommitted(TransGroupId) const override;
    void            PushTransAttribute(TransGroupId, CommonIdRange range,
                                       uint32_t attributeIndex,
                                       TransientData data) override;


    LightGroupId    CreateLightGroup(std::string typeName,
                                     PrimGroupId = TracerConstants::EmptyPrimGroupId) override;
    LightId         ReserveLight(LightGroupId, AttributeCountList,
                                 PrimBatchId = TracerConstants::EmptyPrimBatchId) override;
    LightIdList     ReserveLights(LightGroupId, std::vector<AttributeCountList>,
                                  std::vector<PrimBatchId> = std::vector<PrimBatchId>{}) override;
    void            CommitLightReservations(LightGroupId) override;
    bool            IsLightCommitted(LightGroupId) const override;
    void            PushLightAttribute(LightGroupId, CommonIdRange range,
                                       uint32_t attributeIndex,
                                       TransientData data) override;
    void            PushLightAttribute(LightGroupId, CommonIdRange range,
                                       uint32_t attributeIndex,
                                       TransientData,
                                       std::vector<Optional<TextureId>> textures) override;
    void            PushLightAttribute(LightGroupId, CommonIdRange range,
                                       uint32_t attributeIndex,
                                       std::vector<TextureId>) override;


    CameraGroupId   CreateCameraGroup(std::string typeName) override;
    CameraId        ReserveCamera(CameraGroupId, AttributeCountList) override;
    CameraIdList    ReserveCameras(CameraGroupId, std::vector<AttributeCountList>) override;
    void            CommitCamReservations(CameraGroupId) override;
    bool            IsCamCommitted(CameraGroupId) const override;
    void            PushCamAttribute(CameraGroupId, CommonIdRange range,
                                     uint32_t attributeIndex,
                                     TransientData data) override;


    MediumGroupId   CreateMediumGroup(std::string typeName) override;
    MediumId        ReserveMedium(MediumGroupId, AttributeCountList) override;
    MediumIdList    ReserveMediums(MediumGroupId, std::vector<AttributeCountList>) override;
    void            CommitMediumReservations(MediumGroupId) override;
    bool            IsMediumCommitted(MediumGroupId) const override;
    void            PushMediumAttribute(MediumGroupId, CommonIdRange range,
                                        uint32_t attributeIndex,
                                        TransientData data) override;
    void            PushMediumAttribute(MediumGroupId, CommonIdRange range,
                                        uint32_t attributeIndex,
                                        TransientData,
                                        std::vector<Optional<TextureId>> textures) override;
    void            PushMediumAttribute(MediumGroupId, CommonIdRange range,
                                        uint32_t attributeIndex,
                                        std::vector<TextureId> textures) override;


    SurfaceId           CreateSurface(SurfaceParams) override;
    LightSurfaceId      SetBoundarySurface(LightId, TransformId) override;
    LightSurfaceId      CreateLightSurface(LightSurfaceParams) override;
    CamSurfaceId        CreateCameraSurface(CameraSurfaceParams) override;
    SurfaceCommitResult CommitSurfaces() override;
    CameraTransform     GetCamTransform(CamSurfaceId) const override;

    VolumeId            RegisterVolume(VolumeParams) override;
    VolumeIdList        RegisterVolumes(std::vector<VolumeParams>) override;
    void                SetBoundaryVolume(VolumeId) override;

    RendererId  CreateRenderer(std::string typeName) override;
    void        DestroyRenderer(RendererId) override;
    void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                      TransientData data) override;

    void                SetupRenderEnv(TimelineSemaphore* semaphore,
                                       uint32_t importAlignment,
                                       uint64_t initialAcquireValue) override;
    RenderBufferInfo    StartRender(RendererId, CamSurfaceId,
                                    RenderImageParams,
                                    Optional<uint32_t>,
                                    Optional<uint32_t>) override;
    void                SetCameraTransform(RendererId, CameraTransform) override;
    void                StopRender() override;
    RendererOutput      DoRenderWork() override;

    // Misc
    void                    ClearAll() override;
    void                    Flush() const override;
    void                    SetThreadPool(ThreadPool& tp) override;
    GPUThreadInitFunction   GetThreadInitFunction() const override;
    size_t                  TotalDeviceMemory() const override;
    size_t                  UsedDeviceMemory() const override;
    const TracerParameters& Parameters() const override;
};

inline TracerMock::TracerMock(bool pl)
    : print(pl)
{
    using enum PrimitiveAttributeLogic::E;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    using enum AttributeTexturable;
    using enum AttributeIsColor;

    // =================== //
    //     Primitives      //
    // =================== //
    primMockPack["(P)Triangle"] = PrimMockPack
    {
        .attribInfo = PrimAttributeInfoList
        {
            PrimAttributeInfo(POSITION, MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3),    IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(NORMAL,   MRayDataTypeRT(MRayDataEnum::MR_QUATERNION),  IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(UV0,      MRayDataTypeRT(MRayDataEnum::MR_VECTOR_2),    IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(INDEX,    MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3UI),  IS_SCALAR, MR_MANDATORY)
        },
        .name = "(P)Triangle"
    };

    primMockPack["(P)TriangleSkinned"] = PrimMockPack
    {
        .attribInfo = PrimAttributeInfoList
        {
            PrimAttributeInfo(POSITION,     MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3),    IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(NORMAL,       MRayDataTypeRT(MRayDataEnum::MR_QUATERNION),  IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(UV0,          MRayDataTypeRT(MRayDataEnum::MR_VECTOR_2),    IS_SCALAR, MR_OPTIONAL),
            PrimAttributeInfo(WEIGHT,       MRayDataTypeRT(MRayDataEnum::MR_UNORM_4x8),   IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(WEIGHT_INDEX, MRayDataTypeRT(MRayDataEnum::MR_VECTOR_4UC),  IS_SCALAR, MR_MANDATORY),
            PrimAttributeInfo(INDEX,        MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3UI),  IS_SCALAR, MR_MANDATORY)
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
            MatAttributeInfo("albedo", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR),
            MatAttributeInfo("normalMap", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3),
                             IS_SCALAR, MR_OPTIONAL, MR_TEXTURE_ONLY, IS_PURE_DATA)
        },
        .name = "(Mt)Lambert"
    };
    matMockPack["(Mt)Unreal"] = MatMockPack
    {
        .attribInfo = MatAttributeInfoList
        {
            MatAttributeInfo("albedo", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR),
            MatAttributeInfo("metallic", MRayDataTypeRT(MRayDataEnum::MR_FLOAT),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
            MatAttributeInfo("specular", MRayDataTypeRT(MRayDataEnum::MR_FLOAT),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
            MatAttributeInfo("roughness", MRayDataTypeRT(MRayDataEnum::MR_FLOAT),
                             IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
            MatAttributeInfo("normalMap", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3),
                             IS_SCALAR, MR_OPTIONAL, MR_TEXTURE_ONLY, IS_PURE_DATA)
        },
        .name = "(Mt)Unreal"
    };
    matMockPack["(Mt)Reflect"] = MatMockPack
    {
        .attribInfo = MatAttributeInfoList{},
        .name = "(Mt)Reflect"
    };
    matMockPack["(Mt)Refract"] = MatMockPack
    {
        .attribInfo = MatAttributeInfoList{},
        .name = "(Mt)Refract"
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
            LightAttributeInfo("radiance", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR)
        },
        .name = "(L)Skysphere"
    };
    lightMockPack["(L)Prim(P)Triangle"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR)
        },
        .name ="(L)Prim(P)Triangle"
    };
    lightMockPack["(L)Rectangle"] = LightMockPack
    {
        .attribInfo = LightAttributeInfoList
        {
            LightAttributeInfo("radiance", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                               MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR),
            LightAttributeInfo("position", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                               MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA),
            LightAttributeInfo("right", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                               MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA),
            LightAttributeInfo("up", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                               MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA)
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
            CamAttributeInfo("FovAndPlanes", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_4), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("gaze",         MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("position",     MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR, MR_MANDATORY),
            CamAttributeInfo("up",           MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR, MR_MANDATORY)
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
            TransAttributeInfo("matrix", MRayDataTypeRT(MRayDataEnum::MR_MATRIX_4x4), IS_SCALAR, MR_MANDATORY)
        },
        .name = "(T)Single"
    };
    transMockPack["(T)Multi"] = TransMockPack
    {
        .attribInfo = TransAttributeInfoList
        {
            TransAttributeInfo("matrix", MRayDataTypeRT(MRayDataEnum::MR_MATRIX_4x4), IS_ARRAY, MR_MANDATORY)
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
            MatAttributeInfo("sigmaA", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                             MR_MANDATORY, MR_CONSTANT_ONLY, IS_COLOR),
            MatAttributeInfo("sigmaS", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                             MR_MANDATORY, MR_CONSTANT_ONLY, IS_COLOR),
            MatAttributeInfo("emission", MRayDataTypeRT(MRayDataEnum::MR_VECTOR_3), IS_SCALAR,
                             MR_MANDATORY, MR_CONSTANT_ONLY, IS_COLOR),
            MatAttributeInfo("hgPhase", MRayDataTypeRT(MRayDataEnum::MR_FLOAT), IS_SCALAR,
                             MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA)
        },
        .name = "(Md)Homogeneous"
    };
}

inline TypeNameList TracerMock::PrimitiveGroups() const
{
    using namespace std::string_view_literals;
    return std::vector<std::string_view>
    {
        "(P)Triangle"sv,
        "(P)TriangleSkinned"sv
    };
}

inline TypeNameList TracerMock::MaterialGroups() const
{
    using namespace std::string_view_literals;
    return std::vector<std::string_view>
    {
        "(Mt)Lambert"sv,
        "(Mt)Unreal"sv
    };
}

inline TypeNameList TracerMock::TransformGroups() const
{
    using namespace std::string_view_literals;
    return std::vector<std::string_view>
    {
        "(T)Identity"sv,
        "(T)Single"sv,
        "(T)Multi"sv
    };
}

inline TypeNameList TracerMock::CameraGroups() const
{
    using namespace std::string_view_literals;
    return std::vector<std::string_view>
    {
        "(C)Pinhole"sv
    };
}

inline TypeNameList TracerMock::MediumGroups() const
{
    using namespace std::string_view_literals;
    return std::vector<std::string_view>
    {
        "(Md)Vacuum"sv,
        "(Md)Homogeneous"sv
    };
}

inline TypeNameList TracerMock::LightGroups() const
{
    using namespace std::string_view_literals;
    return std::vector<std::string_view>
    {
        "(L)Null"sv,
        "(L)Skysphere"sv,
        "(L)Rectangle"sv,
        "(L)Prim(P)Triangle"sv
    };
}

inline TypeNameList TracerMock::Renderers() const
{
    return std::vector<std::string_view>{};
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
    std::string_view name;
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(pGLock);
        auto& val = primGroups.at(id);
        atomicCounter = &val.batchCounter;
        name = val.mp.name;
    }

    if(print)
        MRAY_LOG("Reserving primitive over PrimGroup({})[{}], VCount: {}, PCount: {}",
                 static_cast<CommonId>(id), name,
                 count.attributeCount, count.primCount);

    size_t batchId = atomicCounter->fetch_add(1);
    return PrimBatchId(batchId);
}

inline PrimBatchIdList TracerMock::ReservePrimitiveBatches(PrimGroupId id,
                                                           std::vector<PrimCount> primCounts)
{
    std::string_view name;
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(pGLock);
        auto& value = primGroups.at(id);
        atomicCounter = &value.batchCounter;
        name = value.mp.name;
    }

    size_t batchIdFirst = atomicCounter->fetch_add(primCounts.size());
    if(print)
    {
        std::string log;
        for(const auto& count : primCounts)
        {
            log += MRAY_FORMAT("Reserving primitive over PrimGroup({})[{}], "
                               "VCount: {}, PCount: {}\n",
                               static_cast<CommonId>(id), name,
                               count.attributeCount,
                               count.primCount);
        }
        MRAY_LOG("{}", log);
    }

    PrimBatchIdList result;
    result.reserve(primCounts.size());
    for(size_t i = 0; i < primCounts.size(); i++)
    {
        result.push_back(PrimBatchId(i + batchIdFirst));
    };
    return result;
}

inline void TracerMock::CommitPrimReservations(PrimGroupId id)
{
    std::string_view v;
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(pGLock);
        auto& val = primGroups.at(id);
        isCommitted = &val.isComitted;
        v = val.mp.name;
    }

    if(*isCommitted)
        throw MRayError("PrimitiveGroup({})[{}] is already comitted!",
                        static_cast<CommonId>(id), v);
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
    std::lock_guard<std::mutex> lock(pGLock);

    const auto& pg = primGroups.at(gId);
    if(!pg.isComitted)
        throw MRayError("PrimitiveGroup({})[{}] is not committed. "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId), pg.mp.name);
    if(!print) return;

    const auto& attribType = pg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);
    MRAY_LOG("Pushing prim attribute of ({}[{}]:{}),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), pg.mp.name,
             static_cast<CommonId>(batchId),
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushPrimAttribute(PrimGroupId gId,
                                          PrimBatchId batchId,
                                          uint32_t attribIndex,
                                          Vector2ui subBatchRange,
                                          TransientData data)
{
    std::lock_guard<std::mutex> lock(pGLock);

    const auto& pg = primGroups.at(gId);
    if(!pg.isComitted)
        throw MRayError("PrimitiveGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = pg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);
    MRAY_LOG("Pushing prim attribute of ({}[{}]:{})->[{}, {}],"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), pg.mp.name,
             static_cast<CommonId>(batchId),
             subBatchRange[0], subBatchRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}


inline void TracerMock::TransformPrimitives(PrimGroupId,
                                            std::vector<PrimBatchId>,
                                            std::vector<Matrix3x4>)
{
}

inline MatGroupId TracerMock::CreateMaterialGroup(std::string name)
{
    auto loc = matMockPack.find(name);
    if(loc == matMockPack.cend())
        throw MRayError("Failed to find material type: {}", name);

    std::lock_guard<std::mutex> lock(pGLock);
    MatGroupId id = static_cast<MatGroupId>(matGroupCounter++);
    matGroups.try_emplace(id, loc->second, id,
                           std::vector<MaterialId>(),
                           0, false);
    return id;
}

inline MaterialId TracerMock::ReserveMaterial(MatGroupId id, AttributeCountList count)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtGLock);
        atomicCounter = &matGroups.at(id).idCounter;
    }

    if(print)
    {
        std::string attribCountString = "[";
        for(const auto& c : count)
        {
            attribCountString += MRAY_FORMAT("{}, ", c);
        }
        attribCountString += "]";

        MRAY_LOG("Reserving material over MaterialGroup({}), AttribCount: {}",
                 static_cast<CommonId>(id), attribCountString);
    }
    size_t matId = atomicCounter->fetch_add(1);
    return MaterialId(matId);
}

inline MaterialIdList TracerMock::ReserveMaterials(MatGroupId id,
                                                   std::vector<AttributeCountList> countList)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtGLock);
        atomicCounter = &matGroups.at(id).idCounter;
    }

    size_t matIdFirst = atomicCounter->fetch_add(countList.size());
    if(print)
    {
        std::string log;
        for(size_t i = 0; i < countList.size(); i++)
        {
            const AttributeCountList& count = countList[i];
            std::string attribCountString = "[";
            for(const auto& c : count)
            {
                attribCountString += MRAY_FORMAT("{}, ", c);
            }
            attribCountString += "]";

            log += MRAY_FORMAT("Reserving material over MaterialGroup({}), AttribCount: {}\n",
                               static_cast<CommonId>(id), attribCountString);
        }
        MRAY_LOG("{}", log);
    }

    MaterialIdList result;
    result.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        result.push_back(MaterialId(i + matIdFirst));
    };
    return result;
}

inline void TracerMock::CommitMatReservations(MatGroupId id)
{
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtGLock);
        isCommitted = &matGroups.at(id).isComitted;
    }

    if(*isCommitted)
        throw MRayError("MaterialGroup({}) is already comitted!",
                        static_cast<CommonId>(id));
    else
        *isCommitted = true;
}

inline bool TracerMock::IsMatCommitted(MatGroupId id) const
{
    std::lock_guard<std::mutex> lock(mtGLock);
    return matGroups.at(id).isComitted;
}

inline void TracerMock::PushMatAttribute(MatGroupId gId, CommonIdRange matRange,
                                         uint32_t attribIndex,
                                         TransientData data)
{
    std::lock_guard<std::mutex> lock(mtGLock);

    const auto& mg = matGroups.at(gId);
    if(!mg.isComitted)
        throw MRayError("MaterialGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = mg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing mat attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), matRange[0], matRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushMatAttribute(MatGroupId gId, CommonIdRange matRange,
                                         uint32_t attribIndex, TransientData data,
                                         std::vector<Optional<TextureId>>)
{
    std::lock_guard<std::mutex> lock(mtGLock);

    const auto& mg = matGroups.at(gId);
    if(!mg.isComitted)
        throw MRayError("MaterialGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = mg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing mat texturable attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), matRange[0], matRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushMatAttribute(MatGroupId gId, CommonIdRange matRange,
                                         uint32_t,
                                         std::vector<TextureId> textures)
{
    std::lock_guard<std::mutex> lock(mtGLock);
    const auto& mg = matGroups.at(gId);
    if(!mg.isComitted)
        throw MRayError("MaterialGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    std::string textureIdString;
    for(const auto& t : textures)
    {
        textureIdString += MRAY_FORMAT("{}, ", static_cast<CommonId>(t));
    }

    MRAY_LOG("Pushing mat texture attribute of ({}:[{}, {}]),"
             "TextureIds: [{:s}]", static_cast<CommonId>(gId),
             matRange[0], matRange[1], textureIdString);
}

inline TextureId TracerMock::CreateTexture2D(Vector2ui dimensions, uint32_t mipCount,
                                             MRayTextureParameters p)
{
    size_t texId = textureCounter.fetch_add(1);
    if(print)
        MRAY_LOG("Creating Texture2D({}) Dim: ({}, {}), "
                 "Mip:{}, PixelType:{}, ColorSpace:{}, IsColor:{}",
                 static_cast<CommonId>(texId), dimensions[0], dimensions[1],
                 mipCount, MRayPixelTypeStringifier::ToString(p.pixelType.Name()),
                 MRayColorSpaceStringifier::ToString(p.colorSpace),
                 (p.isColor == AttributeIsColor::IS_COLOR) ? true : false);
    return TextureId(texId);
}

inline TextureId TracerMock::CreateTexture3D(Vector3ui dimensions, uint32_t mipCount,
                                             MRayTextureParameters p)
{
    size_t texId = textureCounter.fetch_add(1);
    if(print)
        MRAY_LOG("Creating Texture3D({}) Dim: ({}, {}, {}), "
                 "Mip:{}, PixelType:{}, ColorSpace:{}, IsColor:{}",
                 static_cast<CommonId>(texId), dimensions[0], dimensions[1],
                 dimensions[2], mipCount,
                 MRayPixelTypeStringifier::ToString(p.pixelType.Name()),
                 MRayColorSpaceStringifier::ToString(p.colorSpace),
                 (p.isColor == AttributeIsColor::IS_COLOR) ? true : false);
    return TextureId(texId);
}

inline void TracerMock::CommitTextures()
{
    if(globalTexCommit)
        throw MRayError("Textures are already comitted!");
    globalTexCommit = true;
}

inline void TracerMock::PushTextureData(TextureId tId, uint32_t mipLevel,
                                        TransientData)
{
    if(!globalTexCommit)
        throw MRayError("Textures are not comitted. "
                        "You can not push data to textures!");

    if(!print) return;
    MRAY_LOG("Pushing data to Texture({}) MipLevel:{}",
             static_cast<CommonId>(tId), mipLevel);
}

inline TransGroupId TracerMock::CreateTransformGroup(std::string name)
{
    auto loc = transMockPack.find(name);
    if(loc == transMockPack.cend())
        throw MRayError("Failed to find transform type: {}", name);

    std::lock_guard<std::mutex> lock(tGLock);
    TransGroupId id = static_cast<TransGroupId>(transGroupCounter++);
    transGroups.try_emplace(id, loc->second, id,
                            std::vector<TransformId>(),
                            0, false);
    return id;
}

inline TransformId TracerMock::ReserveTransformation(TransGroupId id, AttributeCountList count)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(tGLock);
        atomicCounter = &transGroups.at(id).idCounter;
    }

    if(print)
    {
        std::string attribCountString = "[";
        for(const auto& c : count)
        {
            attribCountString += MRAY_FORMAT("{}, ", c);
        }
        attribCountString += "]";

        MRAY_LOG("Reserving transform over TransformGroup({}), AttribCount: {}\n",
                 static_cast<CommonId>(id), attribCountString);
    }
    size_t transId = atomicCounter->fetch_add(1);
    return TransformId(transId);
}

inline TransformIdList TracerMock::ReserveTransformations(TransGroupId id,
                                                          std::vector<AttributeCountList> countList)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(tGLock);
        atomicCounter = &transGroups.at(id).idCounter;
    }

    size_t transIdFirst = atomicCounter->fetch_add(countList.size());
    if(print)
    {
        std::string log;
        for(size_t i = 0; i < countList.size(); i++)
        {
            const AttributeCountList& count = countList[i];
            std::string attribCountString = "[";
            for(const auto& c : count)
            {
                attribCountString += MRAY_FORMAT("{}, ", c);
            }
            attribCountString += "]";

            log += MRAY_FORMAT("Reserving transform over TransformGroup({}), AttribCount: {}\n",
                               static_cast<CommonId>(id), attribCountString);
        }
        MRAY_LOG("{}", log);
    }

    TransformIdList result;
    result.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        result.push_back(TransformId(i + transIdFirst));
    };
    return result;
}

inline void TracerMock::CommitTransReservations(TransGroupId id)
{
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(tGLock);
        isCommitted = &transGroups.at(id).isComitted;
    }

    if(*isCommitted)
        throw MRayError("TransformGroup({}) is already comitted!",
                        static_cast<CommonId>(id));
    else
        *isCommitted = true;
}

inline bool TracerMock::IsTransCommitted(TransGroupId id) const
{
    std::lock_guard<std::mutex> lock(tGLock);
    return transGroups.at(id).isComitted;
}

inline void TracerMock::PushTransAttribute(TransGroupId gId, CommonIdRange transRange,
                                           uint32_t attribIndex,
                                           TransientData data)
{
    std::lock_guard<std::mutex> lock(tGLock);

    const auto& tg = transGroups.at(gId);
    if(!tg.isComitted)
        throw MRayError("TransformGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = tg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing trans attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), transRange[0], transRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline LightGroupId TracerMock::CreateLightGroup(std::string name,
                                                 PrimGroupId primGId)
{
    auto loc = lightMockPack.find(name);
    if(loc == lightMockPack.cend())
        throw MRayError("Failed to find light type: {}", name);

    std::lock_guard<std::mutex> lock(lGLock);
    LightGroupId id = static_cast<LightGroupId>(lightGroupCounter++);
    lightGroups.try_emplace(id, loc->second, id, primGId,
                            std::vector<LightId>(),
                            0, false);
    return id;
}

inline LightId TracerMock::ReserveLight(LightGroupId id,
                                        AttributeCountList count,
                                        PrimBatchId primId)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(lGLock);
        atomicCounter = &lightGroups.at(id).idCounter;
    }

    if(print)
    {
        std::string attribCountString = "[";
        for(const auto& c : count)
        {
            attribCountString += MRAY_FORMAT("{}, ", c);
        }
        attribCountString += "]";
        std::string batchString = (primId == TracerConstants::EmptyPrimBatchId)
            ? std::string("Empty")
            : MRAY_FORMAT("{}", static_cast<CommonId>(primId));

        MRAY_LOG("Reserving light over LightGroup({}), AttribCount: {} "
                 "PrimBatchId: {:s}", static_cast<CommonId>(id),
                 attribCountString, batchString);
    }
    size_t lId = atomicCounter->fetch_add(1);
    return LightId(lId);
}

inline LightIdList TracerMock::ReserveLights(LightGroupId id,
                                             std::vector<AttributeCountList> countList,
                                             std::vector<PrimBatchId> primBatches)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(lGLock);
        atomicCounter = &lightGroups.at(id).idCounter;
    }

    size_t lightIdFirst = atomicCounter->fetch_add(countList.size());
    if(print)
    {
        std::string log;
        for(size_t i = 0; i < countList.size(); i++)
        {
            const AttributeCountList& count = countList[i];
            std::string attribCountString = "[";
            for(const auto& c : count)
            {
                attribCountString += MRAY_FORMAT("{}, ", c);
            }
            attribCountString += "]";

            std::string batchString = primBatches.empty()
                ? "Empty"
                : MRAY_FORMAT("{}", static_cast<CommonId>(primBatches[i]));

            log += MRAY_FORMAT("Reserving light over LightGroup({}), AttribCount: {} "
                               "PrimBatchId: {:s}\n", static_cast<CommonId>(id),
                               attribCountString, batchString);
        }
        MRAY_LOG("{}", log);
    }

    LightIdList result;
    result.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        result.push_back(LightId(i + lightIdFirst));
    };
    return result;
}

inline void TracerMock::CommitLightReservations(LightGroupId id)
{
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(lGLock);
        isCommitted = &lightGroups.at(id).isComitted;
    }

    if(*isCommitted)
        throw MRayError("LightGroup({}) is already comitted!",
                        static_cast<CommonId>(id));
    else
        *isCommitted = true;
}

inline bool TracerMock::IsLightCommitted(LightGroupId id) const
{
    std::lock_guard<std::mutex> lock(lGLock);
    return lightGroups.at(id).isComitted;
}

inline void TracerMock::PushLightAttribute(LightGroupId gId, CommonIdRange lightRange,
                                           uint32_t attribIndex,
                                           TransientData data)
{
    std::lock_guard<std::mutex> lock(lGLock);

    const auto& lg = lightGroups.at(gId);
    if(!lg.isComitted)
        throw MRayError("LightGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = lg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing light attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), lightRange[0], lightRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushLightAttribute(LightGroupId gId, CommonIdRange lightRange,
                                           uint32_t attribIndex,
                                           TransientData data,
                                           std::vector<Optional<TextureId>>)
{
    std::lock_guard<std::mutex> lock(lGLock);

    const auto& lg = lightGroups.at(gId);
    if(!lg.isComitted)
        throw MRayError("LightGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = lg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing light texturable attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), lightRange[0], lightRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushLightAttribute(LightGroupId gId, CommonIdRange lightRange,
                                           uint32_t,
                                           std::vector<TextureId> textures)
{
    std::lock_guard<std::mutex> lock(lGLock);
    const auto& lg = lightGroups.at(gId);
    if(!lg.isComitted)
        throw MRayError("LightGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    std::string textureIdString;
    for(const auto& t : textures)
    {
        textureIdString += MRAY_FORMAT("{}, ", static_cast<CommonId>(t));
    }

    MRAY_LOG("Pushing light texture attribute of ({}:[{}, {}]),"
             "TextureIds: [{:s}]", static_cast<CommonId>(gId),
             lightRange[0], lightRange[1], textureIdString);
}

inline CameraGroupId TracerMock::CreateCameraGroup(std::string name)
{
    auto loc = camMockPack.find(name);
    if(loc == camMockPack.cend())
        throw MRayError("Failed to find camera type: {}", name);

    std::lock_guard<std::mutex> lock(cGLock);
    CameraGroupId id = static_cast<CameraGroupId>(camGroupCounter++);
    camGroups.try_emplace(id, loc->second, id,
                          std::vector<CameraId>(),
                          0, false);
    return id;
}

inline CameraId TracerMock::ReserveCamera(CameraGroupId id,
                                          AttributeCountList count)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(cGLock);
        atomicCounter = &camGroups.at(id).idCounter;
    }

    if(print)
    {
        std::string attribCountString = "[";
        for(const auto& c : count)
        {
            attribCountString += MRAY_FORMAT("{}, ", c);
        }
        attribCountString += "]";

        MRAY_LOG("Reserving camera over CameraGroup({}), AttribCount: {}",
                 static_cast<CommonId>(id), attribCountString);
    }
    size_t camId = atomicCounter->fetch_add(1);
    return CameraId(camId);
}

inline CameraIdList TracerMock::ReserveCameras(CameraGroupId id,
                                               std::vector<AttributeCountList> countList)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(cGLock);
        atomicCounter = &camGroups.at(id).idCounter;
    }

    size_t cameraIdFirst = atomicCounter->fetch_add(countList.size());
    if(print)
    {
        std::string log;
        for(size_t i = 0; i < countList.size(); i++)
        {
            const AttributeCountList& count = countList[i];
            std::string attribCountString = "[";
            for(const auto& c : count)
            {
                attribCountString += MRAY_FORMAT("{}, ", c);
            }
            attribCountString += "]";

            log += MRAY_FORMAT("Reserving camera over CameraGroup({}), AttribCount: {}\n",
                               static_cast<CommonId>(id), attribCountString);
        }
        MRAY_LOG("{}", log);
    }

    CameraIdList result;
    result.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        result.push_back(CameraId(i + cameraIdFirst));
    };
    return result;
}

inline void TracerMock::CommitCamReservations(CameraGroupId id)
{
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(cGLock);
        isCommitted = &camGroups.at(id).isComitted;
    }

    if(*isCommitted)
        throw MRayError("CameraGroup({}) is already comitted!",
                        static_cast<CommonId>(id));
    else
        *isCommitted = true;
}

inline bool TracerMock::IsCamCommitted(CameraGroupId id) const
{
    std::lock_guard<std::mutex> lock(cGLock);
    return camGroups.at(id).isComitted;
}

inline void TracerMock::PushCamAttribute(CameraGroupId gId, CommonIdRange camRange,
                                         uint32_t attribIndex,
                                         TransientData data)
{
    std::lock_guard<std::mutex> lock(cGLock);

    const auto& cg = camGroups.at(gId);
    if(!cg.isComitted)
        throw MRayError("CameraGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = cg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing camera attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), camRange[0], camRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline MediumGroupId TracerMock::CreateMediumGroup(std::string name)
{
    auto loc = medMockPack.find(name);
    if(loc == medMockPack.cend())
        throw MRayError("Failed to find medium type: {}", name);

    std::lock_guard<std::mutex> lock(meGLock);
    MediumGroupId id = static_cast<MediumGroupId>(mediumGroupCounter++);
    mediumGroups.try_emplace(id, loc->second, id,
                             std::vector<MediumId>(),
                             0, false);
    return id;
}

inline MediumId TracerMock::ReserveMedium(MediumGroupId id, AttributeCountList count)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(meGLock);
        atomicCounter = &mediumGroups.at(id).idCounter;
    }

    if(print)
    {
        std::string attribCountString = "[";
        for(const auto& c : count)
        {
            attribCountString += MRAY_FORMAT("{}, ", c);
        }
        attribCountString += "]";

        MRAY_LOG("Reserving medium over MediumGroup({}), AttribCount: {}",
                 static_cast<CommonId>(id), attribCountString);
    }
    size_t medId = atomicCounter->fetch_add(1);
    return MediumId(medId);
}

inline MediumIdList TracerMock::ReserveMediums(MediumGroupId id,
                                               std::vector<AttributeCountList> countList)
{
    std::atomic_size_t* atomicCounter = nullptr;
    {
        std::lock_guard<std::mutex> lock(meGLock);
        atomicCounter = &mediumGroups.at(id).idCounter;
    }

    size_t mediumIdFirst = atomicCounter->fetch_add(countList.size());
    if(print)
    {
        std::string log;
        for(size_t i = 0; i < countList.size(); i++)
        {
            const AttributeCountList& count = countList[i];
            std::string attribCountString = "[";
            for(const auto& c : count)
            {
                attribCountString += MRAY_FORMAT("{}, ", c);
            }
            attribCountString += "]";

            log += MRAY_FORMAT("Reserving medium over MediumGroup({}), AttribCount: {}\n",
                               static_cast<CommonId>(id), attribCountString);
        }
        MRAY_LOG("{}", log);
    }

    MediumIdList result;
    result.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        result.push_back(MediumId(i + mediumIdFirst));
    };
    return result;
}

inline void TracerMock::CommitMediumReservations(MediumGroupId id)
{
    bool* isCommitted = nullptr;
    {
        std::lock_guard<std::mutex> lock(meGLock);
        isCommitted = &mediumGroups.at(id).isComitted;
    }

    if(*isCommitted)
        throw MRayError("MediumGroup({}) is already comitted!",
                        static_cast<CommonId>(id));
    else
        *isCommitted = true;
}

inline bool TracerMock::IsMediumCommitted(MediumGroupId id) const
{
    std::lock_guard<std::mutex> lock(meGLock);
    return mediumGroups.at(id).isComitted;
}

inline void TracerMock::PushMediumAttribute(MediumGroupId gId, CommonIdRange mediumRange,
                                            uint32_t attribIndex,
                                            TransientData data)
{
    std::lock_guard<std::mutex> lock(meGLock);

    const auto& mg = mediumGroups.at(gId);
    if(!mg.isComitted)
        throw MRayError("MediumGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = mg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing medium attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), mediumRange[0], mediumRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushMediumAttribute(MediumGroupId gId, CommonIdRange mediumRange,
                                            uint32_t attribIndex,
                                            TransientData data,
                                            std::vector<Optional<TextureId>>)
{
    std::lock_guard<std::mutex> lock(meGLock);

    const auto& mg = mediumGroups.at(gId);
    if(!mg.isComitted)
        throw MRayError("MediumGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    const auto& attribType = mg.mp.attribInfo[attribIndex];
    MRayDataTypeRT dataTypeRT = attribType.dataType;
    MRayDataEnum dataEnum = dataTypeRT.Name();
    size_t dataCount = AcquireTransientDataCount(dataTypeRT, data);

    MRAY_LOG("Pushing medium texturable attribute of ({}:[{}, {}]),"
             "DataType: {:s}, ByteSize: {}",
             static_cast<CommonId>(gId), mediumRange[0], mediumRange[1],
             MRayDataTypeStringifier::ToString(dataEnum), dataCount);
}

inline void TracerMock::PushMediumAttribute(MediumGroupId gId, CommonIdRange mediumRange,
                                            uint32_t,
                                            std::vector<TextureId> textures)
{
    std::lock_guard<std::mutex> lock(meGLock);
    const auto& mg = mediumGroups.at(gId);
    if(!mg.isComitted)
        throw MRayError("MediumGroup({}) is not committed! "
                        "You can not push data to it!",
                        static_cast<CommonId>(gId));
    if(!print) return;

    std::string textureIdString;
    for(const auto& t : textures)
    {
        textureIdString += MRAY_FORMAT("{}, ", static_cast<CommonId>(t));
    }

    MRAY_LOG("Pushing medium texture attribute of ({}:[{}, {}]),"
             "TextureIds: [{:s}]", static_cast<CommonId>(gId),
             mediumRange[0], mediumRange[1], textureIdString);
}

inline SurfaceId TracerMock::CreateSurface(SurfaceParams p)
{
    assert(p.primBatches.size() == p.materials.size());
    assert(p.materials.size() == p.alphaMaps.size());
    assert(p.alphaMaps.size() == p.cullFaceFlags.size());

    size_t surfId = surfaceCounter.fetch_add(1);
    if(!print) return SurfaceId(surfId);

    std::string primIdString;
    std::string matIdString;
    std::string alphaMapString;
    std::string cullFaceString;
    std::string interfaceString;

    for(size_t i = 0; i < p.primBatches.size(); i++)
    {
        primIdString += MRAY_FORMAT("{}, ", static_cast<CommonId>(p.primBatches[i]));
        matIdString += MRAY_FORMAT("{}, ", static_cast<CommonId>(p.materials[i]));
        cullFaceString += MRAY_FORMAT("{}, ", p.cullFaceFlags[i]);
        alphaMapString += (p.alphaMaps[i].has_value())
                            ? MRAY_FORMAT("{}, ", static_cast<CommonId>(p.alphaMaps[i].value()))
                            : "None, ";

        interfaceString += (p.volumes[i] == TracerConstants::InvalidVolume)
                            ? MRAY_FORMAT("{}, ", CommonId(p.volumes[i]))
                            : "None, ";
    }

    MRAY_LOG("Creating Surface({}): Trans:{}, "
             "Prim: [{}], Mat: [{}], AlphaMap: [{}], "
             "CullFace: [{}], Interface: [{}]",
             surfId, static_cast<CommonId>(p.transformId),
             primIdString, matIdString, alphaMapString, cullFaceString,
             interfaceString);
    return SurfaceId(surfId);
}

inline LightSurfaceId TracerMock::SetBoundarySurface(LightId lightId,
                                                     TransformId transformId)
{
    size_t lightSurfId = lightSurfaceCounter.fetch_add(1);
    if(!print) return LightSurfaceId(lightSurfId);

    MRAY_LOG("Setting BoundarySurface({}): Light: {}, Trans: {}",
             lightSurfId,
             static_cast<CommonId>(lightId),
             static_cast<CommonId>(transformId));
    return LightSurfaceId(lightSurfId);
}

inline LightSurfaceId TracerMock::CreateLightSurface(LightSurfaceParams p)
{
    size_t lightSurfId = lightSurfaceCounter.fetch_add(1);
    if(!print) return LightSurfaceId(lightSurfId);

    std::string volumeList;
    for(VolumeId v : p.nestedVolumes)
        volumeList += MRAY_FORMAT("{} ", CommonId(v));

    MRAY_LOG("Creating LightSurface({}): Light: {}, Trans: {}, "
             "Nested in these Volumes: [{}]",
             lightSurfId, static_cast<CommonId>(p.lightId),
             static_cast<CommonId>(p.transformId),
             volumeList);
    return LightSurfaceId(lightSurfId);
}

inline CamSurfaceId TracerMock::CreateCameraSurface(CameraSurfaceParams p)
{
    size_t camSurfId = camSurfaceCounter.fetch_add(1);
    if(!print) return CamSurfaceId(camSurfId);

    std::string volumeList;
    for(VolumeId v : p.nestedVolumes)
        volumeList += MRAY_FORMAT("{} ", CommonId(v));

    MRAY_LOG("Creating CameraSurface({}): Camera: {}, Trans: {}, "
             "Nested in these Volumes: [{}]",
             camSurfId, static_cast<CommonId>(p.cameraId),
             static_cast<CommonId>(p.transformId),
             volumeList);
    return CamSurfaceId(camSurfId);
}

inline VolumeId TracerMock::RegisterVolume(VolumeParams v)
{
    size_t volId = volumeCounter.fetch_add(1);
    if(!print) return VolumeId(volId);

    MRAY_LOG("Creating Volume({}): [Medium: {}, Trans: {}, "
             "Priority: {}]", volId,
             static_cast<CommonId>(v.mediumId),
             static_cast<CommonId>(v.transformId),
             static_cast<CommonId>(v.priority));
    return VolumeId(volId);
}

inline VolumeIdList TracerMock::RegisterVolumes(std::vector<VolumeParams> vols)
{
    size_t volStart = volumeCounter.fetch_add(vols.size());
    VolumeIdList result(vols.size());
    for(size_t i = 0; i < vols.size(); i++)
        result[i] = VolumeId(volStart + i);

    if(!print) return result;

    std::string list = "Bulk Creating Volumes";
    for(size_t i = 0; i < vols.size(); i++)
    {
        list += MRAY_FORMAT("    Volume({}): [Medium: {}, Trans: {}, "
                            "Priority: {}]", size_t(result[i]),
                            static_cast<CommonId>(vols[i].mediumId),
                            static_cast<CommonId>(vols[i].transformId),
                            static_cast<CommonId>(vols[i].priority));
    }
    MRAY_LOG("{}", list);
    return result;
}

inline void TracerMock::SetBoundaryVolume(VolumeId id)
{
    if(!print) return;
    MRAY_LOG("Setting boundary volume ({})", CommonId(id));
}

inline SurfaceCommitResult TracerMock::CommitSurfaces()
{
    if(!print) return SurfaceCommitResult{};

    MRAY_LOG("Committing surfaces");
    return SurfaceCommitResult{};
}

inline CameraTransform TracerMock::GetCamTransform(CamSurfaceId camSurfId) const
{
    MRAY_LOG("Returning transform of {}",
             static_cast<CommonId>(camSurfId));
    return CameraTransform
    {
        .position = Vector3::Zero(),
        .gazePoint = Vector3(0, 0, -1),
        .up = Vector3::YAxis()
    };
}

inline RendererId TracerMock::CreateRenderer(std::string)
{
    throw MRayError("\"CreateRenderer\" is not implemented in mock tracer!");
}

inline void TracerMock::DestroyRenderer(RendererId)
{
    throw MRayError("\"DestroyRenderer\" is not implemented in mock tracer!");
}

inline void TracerMock::PushRendererAttribute(RendererId, uint32_t,
                                              TransientData)
{
    throw MRayError("\"PushRendererAttribute\" is not implemented in mock tracer!");
}

inline void TracerMock::SetupRenderEnv(TimelineSemaphore*,
                                       uint32_t, uint64_t)
{
    throw MRayError("\"SetupRenderEnv\" is not implemented in mock tracer!");
}

inline RenderBufferInfo TracerMock::StartRender(RendererId, CamSurfaceId,
                                                RenderImageParams,
                                                Optional<uint32_t>,
                                                Optional<uint32_t>)
{
    throw MRayError("\"StartRender\" is not implemented in mock tracer!");
}

inline void TracerMock::SetCameraTransform(RendererId, CameraTransform)
{
    throw MRayError("\"SetCameraTransform\" is not implemented in mock tracer!");
}

inline void TracerMock::StopRender()
{
    throw MRayError("\"StopRender\" is not implemented in mock tracer!");
}

inline RendererOutput TracerMock::DoRenderWork()
{
    throw MRayError("\"DoRenderWork\" is not implemented in mock tracer!");
}

inline void TracerMock::ClearAll()
{}

inline void TracerMock::Flush() const
{}

inline void TracerMock::SetThreadPool(ThreadPool&)
{}

inline GPUThreadInitFunction TracerMock::GetThreadInitFunction() const
{
    return nullptr;
}

inline size_t TracerMock::TotalDeviceMemory() const
{
    return 0u;
}

inline size_t TracerMock::UsedDeviceMemory() const
{
    return 0u;
}

inline const TracerParameters& TracerMock::Parameters() const
{
    return params;
}