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
#include "Tracer/RendererC.h"
#include "Tracer/AcceleratorC.h"

namespace BS { class thread_pool; }

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
using RendererGenerator = GeneratorFuncType<RendererI, GPUSystem&>;
using BaseAccelGenerator = GeneratorFuncType<BaseAcceleratorI,
                                            BS::thread_pool&, GPUSystem&,
                                            const AccelGroupGenMap&,
                                            const AccelWorkGenMap&>;

// Type packed surfaces
using MaterialWorkBatchInfo = std::tuple<GenericGroupTransformT*,
                                         GenericGroupMaterialT*,
                                         GenericGroupPrimitiveT*>;
using LightWorkBatchInfo = std::pair<GenericGroupTransformT*,
                                     GenericGroupLightT*>;

using CameraWorkBatchInfo = std::pair<GenericGroupTransformT*,
                                      GenericGroupCameraT*>;

struct WorkBatchInfo
{
    std::vector<CameraWorkBatchInfo>    camWorkBatches;
    std::vector<LightWorkBatchInfo>     lightWorkBatches;
    std::vector<MaterialWorkBatchInfo>  matWorkBatches;
};

struct TypeGeneratorPack
{
    std::map<std::string_view, PrimGenerator>       primGenerator;
    std::map<std::string_view, CamGenerator>        camGenerator;
    std::map<std::string_view, MedGenerator>        medGenerator;
    std::map<std::string_view, MatGenerator>        matGenerator;
    std::map<std::string_view, TransGenerator>      transGenerator;
    std::map<std::string_view, LightGenerator>      lightGenerator;
    std::map<std::string_view, RendererGenerator>   rendererGenerator;
    std::map<AcceleratorType, BaseAccelGenerator>   baseAcceleratorGenerator;
    std::map<AcceleratorType, AccelGroupGenMap>     accelGeneratorMap;
    std::map<AcceleratorType, AccelWorkGenMap>      accelWorkGeneratorMap;
};

// TODO: Move these somewhere safe
template<class K, class V>
class ThreadSafeMap
{
    public:
    using MapType   = std::map<K, V>;
    using Iterator  = typename MapType::iterator;
    private:
    MapType                     map;
    mutable std::shared_mutex   mutex;

    public:
    template<class... Args>
    std::pair<Iterator, bool>   try_emplace(const K& k, Args&&... args);
    const V&                    at(const K& k) const;
    void                        remove_at(const K&);
    void                        clear();

    const MapType&              Map() const;
    MapType&                    Map();
};

template<class T>
class ThreadSafeVector
{
    public:
    using VecType = std::vector<T>;
    using Iterator = typename VecType::iterator;
    private:
    VecType                     vector;
    mutable std::shared_mutex   mutex;

    public:
    template<class... Args>
    T&                          emplace_back(Args&&... args);
    const T&                    at(size_t i) const;
    void                        clear();

    const VecType&              Vec() const;
    VecType&                    Vec();
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
    ThreadSafeMap<RendererId, RendererPtr>           renderers;

    // Surface
    ThreadSafeVector<Pair<SurfaceId, SurfaceParams>>            surfaces;
    ThreadSafeVector<Pair<LightSurfaceId, LightSurfaceParams>>  lightSurfaces;
    ThreadSafeVector<Pair<CamSurfaceId, CameraSurfaceParams>>   cameraSurfaces;
    //
    AcceleratorPtr          accelerator;
    RendererI*              currentRenderer = nullptr;

    std::atomic_uint32_t    primGroupCounter    = 0;
    std::atomic_uint32_t    camGroupCounter     = 0;
    std::atomic_uint32_t    mediumGroupCounter  = 0;
    std::atomic_uint32_t    matGroupCounter     = 0;
    std::atomic_uint32_t    transGroupCounter   = 0;
    std::atomic_uint32_t    lightGroupCounter   = 0;
    std::atomic_uint32_t    redererCounter      = 0;
    // Surface Related
    std::atomic_uint32_t    surfaceCounter      = 0;
    std::atomic_uint32_t    lightSurfaceCounter = 0;
    std::atomic_uint32_t    camSurfaceCounter   = 0;
    // Texture Related
    std::atomic_uint32_t    textureCounter      = 0;
    bool                    globalTexCommit     = false;

    protected:
    BS::thread_pool&    threadPool;
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
    const TypeGeneratorPack&   typeGenerators;

    // Current Types
    std::map<std::string_view, PrimAttributeInfoList>       primAttributeInfoMap;
    std::map<std::string_view, CamAttributeInfoList>        camAttributeInfoMap;
    std::map<std::string_view, MediumAttributeInfoList>     medAttributeInfoMap;
    std::map<std::string_view, MatAttributeInfoList>        matAttributeInfoMap;
    std::map<std::string_view, TransAttributeInfoList>      transAttributeInfoMap;
    std::map<std::string_view, LightAttributeInfoList>      lightAttributeInfoMap;
    std::map<std::string_view, RendererAttributeInfoList>   rendererAttributeInfoMap;

    public:
                        TracerBase(BS::thread_pool&,
                                   const TypeGeneratorPack&);

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


    SurfaceId       CreateSurface(SurfaceParams) override;
    LightSurfaceId  CreateLightSurface(LightSurfaceParams) override;
    CamSurfaceId    CreateCameraSurface(CameraSurfaceParams) override;
    void            CommitSurfaces(AcceleratorType) override;


    RendererId  CreateRenderer(std::string typeName) override;
    void        DestroyRenderer(RendererId) override;
    void        CommitRendererReservations(RendererId) override;
    bool        IsRendererCommitted(RendererId) const override;
    void        PushRendererAttribute(RendererId, uint32_t attributeIndex,
                                      TransientData data) override;

    void            StartRender(RendererId, CamSurfaceId,
                                RenderImageParams) override;
    void            StopRender() override;
    RendererOutput  DoRenderWork() override;

    void            ClearAll() override;
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
void ThreadSafeMap<K, V>::remove_at(const K& k)
{
    std::unique_lock<std::shared_mutex> l(mutex);
    map.erase(map.find(k));
}

template<class K, class V>
void ThreadSafeMap<K, V>::clear()
{
    std::unique_lock<std::shared_mutex> l(mutex);
    map.clear();
}

template<class K, class V>
const typename ThreadSafeMap<K, V>::MapType&
ThreadSafeMap<K, V>::Map() const
{
    return map;
}

template<class K, class V>
typename ThreadSafeMap<K, V>::MapType&
ThreadSafeMap<K, V>::Map()
{
    return map;
}

template<class T>
template<class... Args>
T& ThreadSafeVector<T>::emplace_back(Args&&... args)
{
    std::shared_lock<std::shared_mutex> l(mutex);
    return vector.emplace_back(std::forward<Args>(args)...);
}

template <class T>
const T& ThreadSafeVector<T>::at(size_t i) const
{
    std::shared_lock<std::shared_mutex> l(mutex);
    return vector.at(i);
}

template <class T>
void ThreadSafeVector<T>::clear()
{
    std::unique_lock<std::shared_mutex> l(mutex);
    vector.clear();
}

template <class T>
const typename ThreadSafeVector<T>::VecType&
ThreadSafeVector<T>::Vec() const
{
    return vector;
}

template <class T>
typename ThreadSafeVector<T>::VecType&
ThreadSafeVector<T>::Vec()
{
    return vector;
}