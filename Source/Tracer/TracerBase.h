#pragma once

#include "Core/TracerI.h"
#include "Core/TypeGenFunction.h"
#include "Core/Map.h"

#include "PrimitiveC.h"
#include "MaterialC.h"
#include "MediumC.h"
#include "CameraC.h"
#include "TransformC.h"
#include "LightC.h"
#include "RendererC.h"
#include "AcceleratorC.h"
#include "TextureMemory.h"
#include "Random.h"

class ThreadPool;

using PrimGenerator     = GeneratorFuncType<GenericGroupPrimitiveT, uint32_t,
                                            const GPUSystem&>;
using CamGenerator      = GeneratorFuncType<GenericGroupCameraT, uint32_t,
                                            const GPUSystem&>;
using MedGenerator      = GeneratorFuncType<GenericGroupMediumT, uint32_t,
                                            const GPUSystem&, const TextureViewMap&,
                                            const TextureMap&>;
using MatGenerator      = GeneratorFuncType<GenericGroupMaterialT, uint32_t,
                                            const GPUSystem&, const TextureViewMap&,
                                            const TextureMap&>;
using TransGenerator    = GeneratorFuncType<GenericGroupTransformT, uint32_t,
                                            const GPUSystem&>;
using LightGenerator    = GeneratorFuncType<GenericGroupLightT, uint32_t,
                                            const GPUSystem&, const TextureViewMap&,
                                            const TextureMap&, GenericGroupPrimitiveT&>;
using RendererGenerator = GeneratorFuncType<RendererI,
                                            const RenderImagePtr&,
                                            TracerView,
                                            ThreadPool&,
                                            const GPUSystem&,
                                            const RenderWorkPack&>;
using BaseAccelGenerator = GeneratorFuncType<BaseAcceleratorI,
                                             ThreadPool&,
                                             const GPUSystem&,
                                             const AccelGroupGenMap&,
                                             const AccelWorkGenMap&>;

// Type packed surfaces
using MaterialWorkBatchInfo = Tuple<GenericGroupTransformT*,
                                    GenericGroupMaterialT*,
                                    GenericGroupPrimitiveT*>;
using LightWorkBatchInfo = Tuple<GenericGroupTransformT*,
                                 GenericGroupLightT*>;

using CameraWorkBatchInfo = Tuple<GenericGroupTransformT*,
                                  GenericGroupCameraT*>;

struct WorkBatchInfo
{
    std::vector<CameraWorkBatchInfo>    camWorkBatches;
    std::vector<LightWorkBatchInfo>     lightWorkBatches;
    std::vector<MaterialWorkBatchInfo>  matWorkBatches;
};

static constexpr Pair<LightSurfaceId, LightSurfaceParams> InvalidBoundarySurface =
{
    TracerIdInvalid<LightSurfaceId>,
    LightSurfaceParams{}
};

struct TypeGeneratorPack
{
    Map<std::string_view, PrimGenerator>        primGenerator;
    Map<std::string_view, CamGenerator>         camGenerator;
    Map<std::string_view, MedGenerator>         medGenerator;
    Map<std::string_view, MatGenerator>         matGenerator;
    Map<std::string_view, TransGenerator>       transGenerator;
    Map<std::string_view, LightGenerator>       lightGenerator;
    Map<std::string_view, RendererGenerator>    rendererGenerator;
    Map<std::string_view, RenderWorkPack>       renderWorkGenerator;

    Map<AcceleratorType, BaseAccelGenerator>    baseAcceleratorGenerator;
    Map<AcceleratorType, AccelGroupGenMap>      accelGeneratorMap;
    Map<AcceleratorType, AccelWorkGenMap>       accelWorkGeneratorMap;

    // Attribute Maps
    // TODO: Add more for all other types
    Map<std::string_view, RendererAttributeInfoList> rendererAttribMap;
};

class TracerBase : public TracerI
{
    protected:
    GPUSystem gpuSystem;

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
    Pair<LightSurfaceId, LightSurfaceParams>                    boundarySurface = InvalidBoundarySurface;
    ThreadSafeVector<Pair<SurfaceId, SurfaceParams>>            surfaces;
    ThreadSafeVector<Pair<LightSurfaceId, LightSurfaceParams>>  lightSurfaces;
    ThreadSafeVector<Pair<CamSurfaceId, CameraSurfaceParams>>   cameraSurfaces;
    // Flattened Surface
    std::vector<FlatSurfParams>                                 flattenedSurfaces;
    //
    AcceleratorPtr          accelerator;
    RendererI*              currentRenderer = nullptr;
    RendererId              currentRendererId = TracerIdInvalid<RendererId>;

    std::atomic_uint32_t    primGroupCounter    = 0;
    std::atomic_uint32_t    camGroupCounter     = 0;
    std::atomic_uint32_t    mediumGroupCounter  = 0;
    std::atomic_uint32_t    matGroupCounter     = 0;
    std::atomic_uint32_t    transGroupCounter   = 0;
    std::atomic_uint32_t    lightGroupCounter   = 0;
    std::atomic_uint32_t    rendererCounter      = 0;
    // Surface Related
    std::atomic_uint32_t    surfaceCounter      = 0;
    std::atomic_uint32_t    lightSurfaceCounter = 0;
    std::atomic_uint32_t    camSurfaceCounter   = 0;

    protected:
    ThreadPool*         globalThreadPool;

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
    // This is common type generator so this is separate
    FilterGeneratorMap  filterGenMap;
    //
    RNGGeneratorMap     rngGenMap;
    // Loaded Parameters
    TracerParameters    params;
    // Texture Related
    TextureMemory       texMem;
    // Render Framebuffer
    RenderImagePtr      renderImage;

    // Deduplicated Volume List
    std::vector<VolumeKeyPack> globalVolumeList;

    // Current Types
    Map<std::string_view, PrimAttributeInfoList>       primAttributeInfoMap;
    Map<std::string_view, CamAttributeInfoList>        camAttributeInfoMap;
    Map<std::string_view, MediumAttributeInfoList>     medAttributeInfoMap;
    Map<std::string_view, MatAttributeInfoList>        matAttributeInfoMap;
    Map<std::string_view, TransAttributeInfoList>      transAttributeInfoMap;
    Map<std::string_view, LightAttributeInfoList>      lightAttributeInfoMap;
    //Map<std::string_view, RendererAttributeInfoList>   rendererAttributeInfoMap;

    void            PopulateAttribInfoAndTypeLists();
    TracerView      GenerateTracerView();
    void            GenerateDefaultGroups();
    void            GenerateGlobalVolumeList();

    public:
    // Constructors & Destructor
                    TracerBase(const TypeGeneratorPack&,
                               const TracerParameters& tracerParams);
                    TracerBase(const TracerBase&) = delete;
                    TracerBase(TracerBase&&) = delete;
    TracerBase&     operator=(const TracerBase&) = delete;
    TracerBase&     operator=(TracerBase&&) = delete;
                    ~TracerBase() = default;


    TypeNameList    PrimitiveGroups() const override;
    TypeNameList    MaterialGroups() const override;
    TypeNameList    TransformGroups() const override;
    TypeNameList    CameraGroups() const override;
    TypeNameList    MediumGroups() const override;
    TypeNameList    LightGroups() const override;
    TypeNameList    Renderers() const override;

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
    void            TransformPrimitives(PrimGroupId,
                                        std::vector<PrimBatchId>,
                                        std::vector<Matrix4x4>) override;

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
    LightSurfaceId      SetBoundarySurface(LightSurfaceParams) override;
    LightSurfaceId      CreateLightSurface(LightSurfaceParams) override;
    CamSurfaceId        CreateCameraSurface(CameraSurfaceParams) override;
    SurfaceCommitResult CommitSurfaces() override;
    CameraTransform     GetCamTransform(CamSurfaceId) const override;

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

    void                    ClearAll() override;
    void                    Flush() const override;
    GPUThreadInitFunction   GetThreadInitFunction() const override;
    void                    SetThreadPool(ThreadPool&) override;
    size_t                  TotalDeviceMemory() const override;
    size_t                  UsedDeviceMemory() const override;
    const TracerParameters& Parameters() const override;
};