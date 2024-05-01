#pragma once

#include <map>
#include <shared_mutex>

#include "Core/TracerI.h"
#include "Core/TypeGenFunction.h"

#include "Tracer/PrimitiveC.h"
#include "Tracer/MaterialC.h"
#include "Tracer/MediumC.h"
#include "Tracer/CameraC.h"
#include "Tracer/TransformC.h"
#include "Tracer/LightC.h"

class RendererI
{
    public:
    RendererAttributeInfoList AttributeInfo() const
    {
        return {};
    };

    std::string_view Name() const
    {
        using namespace std::string_view_literals;
        static constexpr auto R = "TEST"sv;
        return R;
    }
};

using PrimGroupPtr      = std::unique_ptr<GenericGroupPrimitiveT>;
using CameraGroupPtr    = std::unique_ptr<GenericGroupCameraT>;
using MediumGroupPtr    = std::unique_ptr<GenericGroupMediumT>;
using MaterialGroupPtr  = std::unique_ptr<GenericGroupMaterialT>;
using TransformGroupPtr = std::unique_ptr<GenericGroupTransformT>;
using LightGroupPtr     = std::unique_ptr<GenericGroupLightT>;
using RendererPtr       = std::unique_ptr<RendererI>;

//using XX = std::map<std::string_view, PrimGroupGenerator>;

using PrimGenerator     = GeneratorFuncType<GenericGroupPrimitiveT, uint32_t,
                                            GPUSystem&>;
using CamGenerator      = GeneratorFuncType<GenericGroupCameraT, uint32_t,
                                            GPUSystem&>;
using MedGenerator      = GeneratorFuncType<GenericGroupMediumT, uint32_t,
                                            GPUSystem&, const TextureViewMap&>;
using MatGenerator      = GeneratorFuncType<GenericGroupMaterialT, uint32_t,
                                            GPUSystem&, const TextureViewMap&>;
using TransGenerator    = GeneratorFuncType<GenericGroupTransformT, uint32_t,
                                            GPUSystem&>;
using LightGenerator    = GeneratorFuncType<GenericGroupLightT, uint32_t,
                                            GPUSystem&, const TextureViewMap&,
                                            GenericGroupPrimitiveT&>;
using RendererGenerator = GeneratorFuncType<RendererI>;

// TODO: Move this somewhere safe
template<class K, class V>
class ThreadSafeMap
{
    public:
    using MapType = std::map<K, V>;
    using Iterator = typename MapType::iterator;
    private:
    std::map<K, V>              map;
    mutable std::shared_mutex   mutex;

    public:
    template<class... Args>
    std::pair<Iterator, bool>   try_emplace(const K& k, Args&&... args);
    const V&                    at(const K& k) const;
    void                        clear();
};

class TracerBase : public TracerI
{
    private:
    // Instantiated Types
    ThreadSafeMap<PrimGroupId, PrimGroupPtr>         primGroups;
    ThreadSafeMap<CameraGroupId, CameraGroupPtr>     camGroups;
    ThreadSafeMap<MediumGroupId, MediumGroupPtr>     mediumGroups;
    ThreadSafeMap<MatGroupId, MaterialGroupPtr>      matGroups;
    ThreadSafeMap<TransGroupId, TransformGroupPtr>   transGroups;
    ThreadSafeMap<LightGroupId, LightGroupPtr>       lightGroups;
    ThreadSafeMap<RendererId, RendererPtr>           rendererGroups;
    // Current Renderer
    RendererPtr currentRenderer;

    std::atomic_uint32_t    primGroupCounter    = 0;
    std::atomic_uint32_t    camGroupCounter     = 0;
    std::atomic_uint32_t    mediumGroupCounter  = 0;
    std::atomic_uint32_t    matGroupCounter     = 0;
    std::atomic_uint32_t    transGroupCounter   = 0;
    std::atomic_uint32_t    lightGroupCounter   = 0;
    // Surface Related
    std::atomic_uint32_t    surfaceCounter      = 0;
    std::atomic_uint32_t    lightSurfaceCounter = 0;
    std::atomic_uint32_t    camSurfaceCounter   = 0;
    // Texture Related
    std::atomic_uint32_t    textureCounter      = 0;
    bool                    globalTexCommit     = false;

    protected:
    GPUSystem           gpuSystem;
    TextureViewMap      texViewMap;

    // Supported Types
    TypeNameList        primTypes;
    TypeNameList        camTypes;
    TypeNameList        medTypes;
    TypeNameList        matTypes;
    TypeNameList        transTypes;
    TypeNameList        lightTypes;
    TypeNameList        rendererTypes;

    // Type Generators
    std::map<std::string_view, PrimGenerator>       primGenerator;
    std::map<std::string_view, CamGenerator>        camGenerator;
    std::map<std::string_view, MedGenerator>        medGenerator;
    std::map<std::string_view, MatGenerator>        matGenerator;
    std::map<std::string_view, TransGenerator>      transGenerator;
    std::map<std::string_view, LightGenerator>      lightGenerator;
    std::map<std::string_view, RendererGenerator>   rendererGenerator;

    // Type Generators
    std::map<std::string_view, PrimAttributeInfoList>       primAttributeInfoMap;
    std::map<std::string_view, CamAttributeInfoList>        camAttributeInfoMap;
    std::map<std::string_view, MediumAttributeInfoList>     medAttributeInfoMap;
    std::map<std::string_view, MatAttributeInfoList>        matAttributeInfoMap;
    std::map<std::string_view, TransAttributeInfoList>      transAttributeInfoMap;
    std::map<std::string_view, LightAttributeInfoList>      lightAttributeInfoMap;
    std::map<std::string_view, RendererAttributeInfoList>   rendererAttributeInfoMap;

    public:
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
                                    MRayPixelEnum pixelType,
                                    AttributeIsColor) override;
    TextureId       CreateTexture3D(Vector3ui size, uint32_t mipCount,
                                    MRayPixelEnum pixelType,
                                    AttributeIsColor) override;
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


    LightGroupId    CreateLightGroup(std::string typeName,
                                     PrimGroupId = TracerConstants::EmptyPrimitive) override;
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
    void        DestroyRenderer(RendererId) override;
    void        CommitRendererReservations(RendererId) override;
    bool        IsRendererCommitted(RendererId) const override;
    void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                      TransientData data) override;

    void        StartRender(RendererId, CamSurfaceId,
                            RenderImageParams) override;
    void        StopRender() override;
    //
    Optional<TracerImgOutput> DoRenderWork() override;

    void        ClearAll() override;
};

template<class K, class V>
template<class... Args>
std::pair<typename ThreadSafeMap<K, V>::Iterator, bool>
ThreadSafeMap<K, V>::try_emplace(const K& k, Args&&... args)
{
    std::unique_lock<std::shared_mutex> l(mutex);
    return map.try_emplace(k, std::forward<Args>(args)...);
}

template<class K, class V>
const V& ThreadSafeMap<K, V>::at(const K& k) const
{
    std::shared_lock<std::shared_mutex> l(mutex);
    return map.at(k);
}

template<class K, class V>
void ThreadSafeMap<K, V>::clear()
{
    std::unique_lock<std::shared_mutex> l(mutex);
    map.clear();
}