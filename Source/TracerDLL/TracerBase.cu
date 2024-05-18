#include "TracerBase.h"
#include <BS/BS_thread_pool.hpp>

TracerBase::TracerBase(BS::thread_pool& tp,
                       const TypeGeneratorPack& tGen)
    : threadPool(tp)
    , typeGenerators(tGen)
{
    // Inject the CUDA "setDevice()" equavilent to the threads
    // to initialize GPU usage
    BS::concurrency_t threadCount = threadPool.get_thread_count();
    threadPool.reset(threadCount, gpuSystem.GetThreadInitFunction());
}

TypeNameList TracerBase::PrimitiveGroups() const
{
    return primTypes;
}

TypeNameList TracerBase::MaterialGroups() const
{
    return matTypes;
}

TypeNameList TracerBase::TransformGroups() const
{
    return transTypes;
}

TypeNameList TracerBase::CameraGroups() const
{
    return camTypes;
}

TypeNameList TracerBase::MediumGroups() const
{
    return medTypes;
}

TypeNameList TracerBase::LightGroups() const
{
    return lightTypes;
}

PrimAttributeInfoList TracerBase::AttributeInfo(PrimGroupId id) const
{
    return primGroups.at(id)->AttributeInfo();
}

CamAttributeInfoList TracerBase::AttributeInfo(CameraGroupId id) const
{
    return camGroups.at(id)->AttributeInfo();
}

MediumAttributeInfoList TracerBase::AttributeInfo(MediumGroupId id) const
{
    return mediumGroups.at(id)->AttributeInfo();
}

MatAttributeInfoList TracerBase::AttributeInfo(MatGroupId id) const
{
    return matGroups.at(id)->AttributeInfo();
}

TransAttributeInfoList TracerBase::AttributeInfo(TransGroupId id) const
{
    return transGroups.at(id)->AttributeInfo();
}

LightAttributeInfoList TracerBase::AttributeInfo(LightGroupId id) const
{
    return lightGroups.at(id)->AttributeInfo();
}

RendererAttributeInfoList TracerBase::AttributeInfo(RendererId id) const
{
    return renderers.at(id)->AttributeInfo();
}

PrimAttributeInfoList TracerBase::AttributeInfoPrim(std::string_view name) const
{
    return primAttributeInfoMap.at(name);
}

CamAttributeInfoList TracerBase::AttributeInfoCam(std::string_view name) const
{
    return camAttributeInfoMap.at(name);
}

MediumAttributeInfoList TracerBase::AttributeInfoMedium(std::string_view name) const
{
    return medAttributeInfoMap.at(name);
}

MatAttributeInfoList TracerBase::AttributeInfoMat(std::string_view name) const
{
    return matAttributeInfoMap.at(name);
}

TransAttributeInfoList TracerBase::AttributeInfoTrans(std::string_view name) const
{
    return transAttributeInfoMap.at(name);
}

LightAttributeInfoList TracerBase::AttributeInfoLight(std::string_view name) const
{
    return lightAttributeInfoMap.at(name);
}

RendererAttributeInfoList TracerBase::AttributeInfoRenderer(std::string_view name) const
{
    return rendererAttributeInfoMap.at(name);
}

std::string TracerBase::TypeName(PrimGroupId id) const
{
    return std::string(primGroups.at(id)->Name());
}

std::string TracerBase::TypeName(CameraGroupId id) const
{
    return std::string(camGroups.at(id)->Name());
}

std::string TracerBase::TypeName(MediumGroupId id) const
{
    return std::string(mediumGroups.at(id)->Name());
}

std::string TracerBase::TypeName(MatGroupId id) const
{
    return std::string(matGroups.at(id)->Name());
}

std::string TracerBase::TypeName(TransGroupId id) const
{
    return std::string(transGroups.at(id)->Name());
}

std::string TracerBase::TypeName(LightGroupId id) const
{
    return std::string(lightGroups.at(id)->Name());
}

std::string TracerBase::TypeName(RendererId id) const
{
    return std::string(renderers.at(id)->Name());
}

PrimGroupId TracerBase::CreatePrimitiveGroup(std::string name)
{
    const auto& genFunc = typeGenerators.primGenerator.at(name);

    uint32_t idInt = primGroupCounter.fetch_add(1);
    PrimGroupId id = static_cast<PrimGroupId>(idInt);
    primGroups.try_emplace(id, genFunc(static_cast<uint32_t>(id),
                                       gpuSystem));
    return id;
}

PrimBatchId TracerBase::ReservePrimitiveBatch(PrimGroupId id, PrimCount count)
{
    std::vector<AttributeCountList> input = {{count.primCount, count.attributeCount}};
    std::vector<PrimBatchKey> output;
    output = primGroups.at(id)->Reserve(input);
    using T = typename PrimBatchKey::Type;
    return PrimBatchId(static_cast<T>(output.front()));
}

PrimBatchIdList TracerBase::ReservePrimitiveBatches(PrimGroupId id,
                                                    std::vector<PrimCount> primCounts)
{
    std::vector<AttributeCountList> input;
    input.reserve(primCounts.size());
    for(const auto& pc : primCounts)
        input.push_back({pc.primCount, pc.attributeCount});

    std::vector<PrimBatchKey> output;
    output = primGroups.at(id)->Reserve(input);

    PrimBatchIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        using T = typename PrimBatchKey::Type;
        result.push_back(PrimBatchId(static_cast<T>(key)));
    }
    return result;
}

void TracerBase::CommitPrimReservations(PrimGroupId id)
{
    primGroups.at(id)->CommitReservations();
}

bool TracerBase::IsPrimCommitted(PrimGroupId id) const
{
    return primGroups.at(id)->IsInCommitState();
}

void TracerBase::PushPrimAttribute(PrimGroupId gId,
                                   PrimBatchId batchId,
                                   uint32_t attribIndex,
                                   TransientData data)
{
    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);

    PrimBatchKey key(static_cast<uint32_t>(batchId));
    primGroups.at(gId)->PushAttribute(key, attribIndex,
                                      std::move(data),
                                      queue);
}

void TracerBase::PushPrimAttribute(PrimGroupId gId,
                                   PrimBatchId batchId,
                                   uint32_t attribIndex,
                                   Vector2ui subBatchRange,
                                   TransientData data)
{
    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);

    PrimBatchKey key(static_cast<uint32_t>(batchId));
    primGroups.at(gId)->PushAttribute(key, attribIndex,
                                      subBatchRange,
                                      std::move(data),
                                      queue);
}

MatGroupId TracerBase::CreateMaterialGroup(std::string name)
{
    const auto& genFunc = typeGenerators.matGenerator.at(name);
    uint32_t idInt = matGroupCounter.fetch_add(1);
    MatGroupId id = static_cast<MatGroupId>(idInt);
    matGroups.try_emplace(id, genFunc(static_cast<uint32_t>(id),
                                      gpuSystem, texViewMap));
    return id;
}

MaterialId TracerBase::ReserveMaterial(MatGroupId id,
                                       AttributeCountList count,
                                       MediumPair mediumPair)
{
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    MediumKeyPairList medPairList;
    medPairList.reserve(1);
    using MedKT = typename MediumKey::Type;
    medPairList.emplace_back(MediumKey(static_cast<MedKT>(mediumPair.first)),
                             MediumKey(static_cast<MedKT>(mediumPair.second)));

    std::vector<MaterialKey> output;
    output = matGroups.at(id)->Reserve(attribCountList, medPairList);

    using MatKT = typename MaterialKey::Type;
    MaterialId result = MaterialId(static_cast<MatKT>(output.front()));
    return result;
}

MaterialIdList TracerBase::ReserveMaterials(MatGroupId id,
                                            std::vector<AttributeCountList> countList,
                                            std::vector<MediumPair> medPairs)
{
    assert(medPairs.size() == countList.size());
    MediumKeyPairList medPairList;
    medPairList.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        using MedKT = typename MediumKey::Type;
        medPairList.emplace_back(MediumKey(static_cast<MedKT>(medPairs[i].first)),
                                 MediumKey(static_cast<MedKT>(medPairs[i].second)));
    }

    std::vector<MaterialKey> output;
    output = matGroups.at(id)->Reserve(countList, medPairList);

    MaterialIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        using T = typename MaterialKey::Type;
        result.push_back(MaterialId(static_cast<T>(key)));
    }
    return result;
}

void TracerBase::CommitMatReservations(MatGroupId id)
{
    matGroups.at(id)->CommitReservations();
}

bool TracerBase::IsMatCommitted(MatGroupId id) const
{
    return matGroups.at(id)->IsInCommitState();
}

void TracerBase::PushMatAttribute(MatGroupId gId, Vector2ui matRange,
                                  uint32_t attribIndex,
                                  TransientData data)
{
    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MaterialKey::Type;
    auto keyStart = MaterialKey(static_cast<T>(matRange[0]));
    auto keyEnd = MaterialKey(static_cast<T>(matRange[1]));
    matGroups.at(gId)->PushAttribute(keyStart, keyEnd, attribIndex,
                                     std::move(data), queue);
}

void TracerBase::PushMatAttribute(MatGroupId gId, Vector2ui matRange,
                                  uint32_t attribIndex, TransientData data,
                                  std::vector<Optional<TextureId>> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MaterialKey::Type;
    auto keyStart = MaterialKey(static_cast<T>(matRange[0]));
    auto keyEnd = MaterialKey(static_cast<T>(matRange[1]));
    matGroups.at(gId)->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                        std::move(data),
                                        std::move(textures), queue);
}

void TracerBase::PushMatAttribute(MatGroupId gId, Vector2ui matRange,
                                  uint32_t attribIndex,
                                  std::vector<TextureId> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MaterialKey::Type;
    auto keyStart = MaterialKey(static_cast<T>(matRange[0]));
    auto keyEnd = MaterialKey(static_cast<T>(matRange[1]));
    matGroups.at(gId)->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                        std::move(textures), queue);
}

void TracerBase::CommitTexColorSpace(MRayColorSpaceEnum)
{
    throw MRayError("Textures are not impl.");
}

TextureId TracerBase::CreateTexture2D(Vector2ui, uint32_t,
                                      MRayPixelEnum,
                                      AttributeIsColor)
{
    throw MRayError("Textures are not impl.");
}

TextureId TracerBase::CreateTexture3D(Vector3ui, uint32_t,
                                      MRayPixelEnum,
                                      AttributeIsColor)
{
    throw MRayError("Textures are not impl.");
}

MRayDataTypeRT TracerBase::GetTexturePixelType(TextureId) const
{
    throw MRayError("\"GetTexturePixelType\" is not implemented in mock tracer!");
}

void TracerBase::CommitTextures()
{
    throw MRayError("Textures are not impl.");
}

void TracerBase::PushTextureData(TextureId, uint32_t,
                                 TransientData)
{
    throw MRayError("Textures are not impl.");
}

TransGroupId TracerBase::CreateTransformGroup(std::string name)
{
    const auto& genFunc = typeGenerators.transGenerator.at(name);
    uint32_t idInt = transGroupCounter.fetch_add(1);
    TransGroupId id = static_cast<TransGroupId>(idInt);
    transGroups.try_emplace(id, genFunc(static_cast<uint32_t>(id), gpuSystem));
    return id;
}

TransformId TracerBase::ReserveTransformation(TransGroupId id, AttributeCountList count)
{
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<TransformKey> output;
    output = transGroups.at(id)->Reserve(attribCountList);

    using T = typename TransformKey::Type;
    TransformId result = TransformId(static_cast<T>(output.front()));
    return result;
}

TransformIdList TracerBase::ReserveTransformations(TransGroupId id,
                                                   std::vector<AttributeCountList> countList)
{
    std::vector<TransformKey> output;
    output = transGroups.at(id)->Reserve(countList);

    TransformIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        using T = typename TransformKey::Type;
        result.push_back(TransformId(static_cast<T>(key)));
    }
    return result;
}

void TracerBase::CommitTransReservations(TransGroupId id)
{
    transGroups.at(id)->CommitReservations();
}

bool TracerBase::IsTransCommitted(TransGroupId id) const
{
    return transGroups.at(id)->IsInCommitState();
}

void TracerBase::PushTransAttribute(TransGroupId gId, Vector2ui transRange,
                                    uint32_t attribIndex,
                                    TransientData data)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename TransformKey::Type;
    auto keyStart = TransformKey(static_cast<T>(transRange[0]));
    auto keyEnd = TransformKey(static_cast<T>(transRange[1]));
    transGroups.at(gId)->PushAttribute(keyStart, keyEnd, attribIndex,
                                       std::move(data), queue);
}

LightGroupId TracerBase::CreateLightGroup(std::string name,
                                          PrimGroupId primGId)
{
    const auto& genFunc = typeGenerators.lightGenerator.at(name);
    uint32_t idInt = lightGroupCounter.fetch_add(1);
    LightGroupId id = static_cast<LightGroupId>(idInt);
    GenericGroupPrimitiveT& primGroupPtr = *primGroups.at(primGId).get();
    lightGroups.try_emplace(id, genFunc(static_cast<uint32_t>(id), gpuSystem,
                                        texViewMap, primGroupPtr));
    return id;
}

LightId TracerBase::ReserveLight(LightGroupId id,
                                 AttributeCountList count,
                                 PrimBatchId primId)
{
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    PrimBatchList primList;
    primList.reserve(1);
    using PrimBatchKT = typename PrimBatchKey::Type;
    primList.emplace_back(PrimBatchKey(static_cast<PrimBatchKT>(primId)));

    std::vector<LightKey> output;
    output = lightGroups.at(id)->Reserve(attribCountList, primList);

    using LightKT = typename LightKey::Type;
    LightId result = LightId(static_cast<LightKT>(output.front()));
    return result;
}

LightIdList TracerBase::ReserveLights(LightGroupId id,
                                      std::vector<AttributeCountList> countList,
                                      std::vector<PrimBatchId> primBatches)
{
    assert(primBatches.size() == countList.size());
    PrimBatchList primList;
    primList.reserve(countList.size());
    for(size_t i = 0; i < countList.size(); i++)
    {
        using PrimBatchKT = typename PrimBatchKey::Type;
        primList.emplace_back(PrimBatchKey(static_cast<PrimBatchKT>(primBatches[i])));
    }

    std::vector<LightKey> output;
    output = lightGroups.at(id)->Reserve(countList, primList);

    LightIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        using T = typename LightKey::Type;
        result.push_back(LightId(static_cast<T>(key)));
    }
    return result;
}

void TracerBase::CommitLightReservations(LightGroupId id)
{
    lightGroups.at(id)->CommitReservations();
}

bool TracerBase::IsLightCommitted(LightGroupId id) const
{
    return lightGroups.at(id)->IsInCommitState();
}

void TracerBase::PushLightAttribute(LightGroupId gId, Vector2ui lightRange,
                                    uint32_t attribIndex,
                                    TransientData data)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename LightKey::Type;
    auto keyStart = LightKey(static_cast<T>(lightRange[0]));
    auto keyEnd = LightKey(static_cast<T>(lightRange[1]));
    lightGroups.at(gId)->PushAttribute(keyStart, keyEnd, attribIndex,
                                       std::move(data), queue);
}

void TracerBase::PushLightAttribute(LightGroupId gId, Vector2ui lightRange,
                                    uint32_t attribIndex,
                                    TransientData data,
                                    std::vector<Optional<TextureId>> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename LightKey::Type;
    auto keyStart = LightKey(static_cast<T>(lightRange[0]));
    auto keyEnd = LightKey(static_cast<T>(lightRange[1]));
    lightGroups.at(gId)->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(data), std::move(textures),
                                          queue);
}

void TracerBase::PushLightAttribute(LightGroupId gId, Vector2ui lightRange,
                                    uint32_t attribIndex,
                                    std::vector<TextureId> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename LightKey::Type;
    auto keyStart = LightKey(static_cast<T>(lightRange[0]));
    auto keyEnd = LightKey(static_cast<T>(lightRange[1]));
    lightGroups.at(gId)->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(textures), queue);
}

CameraGroupId TracerBase::CreateCameraGroup(std::string name)
{
    const auto& genFunc = typeGenerators.camGenerator.at(name);
    uint32_t idInt = camGroupCounter.fetch_add(1);
    CameraGroupId id = static_cast<CameraGroupId>(idInt);
    camGroups.try_emplace(id, genFunc(static_cast<uint32_t>(id), gpuSystem));
    return id;
}

CameraId TracerBase::ReserveCamera(CameraGroupId id,
                                   AttributeCountList count)
{
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<CameraKey> output;
    output = camGroups.at(id)->Reserve(attribCountList);

    using T = typename CameraKey::Type;
    CameraId result = CameraId(static_cast<T>(output.front()));
    return result;
}

CameraIdList TracerBase::ReserveCameras(CameraGroupId id,
                                        std::vector<AttributeCountList> countList)
{
    std::vector<CameraKey> output;
    output = camGroups.at(id)->Reserve(countList);

    CameraIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        using T = typename CameraKey::Type;
        result.push_back(CameraId(static_cast<T>(key)));
    }
    return result;
}

void TracerBase::CommitCamReservations(CameraGroupId id)
{
    camGroups.at(id)->CommitReservations();
}

bool TracerBase::IsCamCommitted(CameraGroupId id) const
{
    return camGroups.at(id)->IsInCommitState();
}

void TracerBase::PushCamAttribute(CameraGroupId gId, Vector2ui camRange,
                                  uint32_t attribIndex,
                                  TransientData data)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename CameraKey::Type;
    auto keyStart = CameraKey(static_cast<T>(camRange[0]));
    auto keyEnd = CameraKey(static_cast<T>(camRange[1]));
    camGroups.at(gId)->PushAttribute(keyStart, keyEnd, attribIndex,
                                     std::move(data), queue);
}

MediumGroupId TracerBase::CreateMediumGroup(std::string name)
{
    const auto& genFunc = typeGenerators.medGenerator.at(name);
    uint32_t idInt = mediumGroupCounter.fetch_add(1);
    MediumGroupId id = static_cast<MediumGroupId>(idInt);
    mediumGroups.try_emplace(id, genFunc(static_cast<uint32_t>(id), gpuSystem, texViewMap));
    return id;
}

MediumId TracerBase::ReserveMedium(MediumGroupId id, AttributeCountList count)
{
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<MediumKey> output;
    output = mediumGroups.at(id)->Reserve(attribCountList);

    using T = typename MediumKey::Type;
    MediumId result = MediumId(static_cast<T>(output.front()));
    return result;
}

MediumIdList TracerBase::ReserveMediums(MediumGroupId id,
                                        std::vector<AttributeCountList> countList)
{
    std::vector<MediumKey> output;
    output = mediumGroups.at(id)->Reserve(countList);
    MediumIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        using T = typename MediumKey::Type;
        result.push_back(MediumId(static_cast<T>(key)));
    }
    return result;
}

void TracerBase::CommitMediumReservations(MediumGroupId id)
{
    mediumGroups.at(id)->CommitReservations();
}

bool TracerBase::IsMediumCommitted(MediumGroupId id) const
{
    return mediumGroups.at(id)->IsInCommitState();
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, Vector2ui mediumRange,
                                     uint32_t attribIndex,
                                     TransientData data)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MediumKey::Type;
    auto keyStart = MediumKey(static_cast<T>(mediumRange[0]));
    auto keyEnd = MediumKey(static_cast<T>(mediumRange[1]));
    mediumGroups.at(gId)->PushAttribute(keyStart, keyEnd, attribIndex,
                                        std::move(data), queue);
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, Vector2ui mediumRange,
                                     uint32_t attribIndex,
                                     TransientData data,
                                     std::vector<Optional<TextureId>> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MediumKey::Type;
    auto keyStart = MediumKey(static_cast<T>(mediumRange[0]));
    auto keyEnd = MediumKey(static_cast<T>(mediumRange[1]));
    mediumGroups.at(gId)->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                        std::move(data), std::move(textures), queue);
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, Vector2ui mediumRange,
                                     uint32_t attribIndex,
                                     std::vector<TextureId> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MediumKey::Type;
    auto keyStart = MediumKey(static_cast<T>(mediumRange[0]));
    auto keyEnd = MediumKey(static_cast<T>(mediumRange[1]));
    mediumGroups.at(gId)->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                           std::move(textures), queue);
}

SurfaceId TracerBase::CreateSurface(SurfaceParams p)
{
    // Validate that all batches are in the same group
    uint32_t groupId = 0;
    uint32_t groupIdFirst = 0;
    for(size_t i = 0; i < p.primBatches.size(); i++)
    {
        PrimBatchKey key(static_cast<uint32_t>(p.primBatches[i]));
        uint32_t currId = key.FetchBatchPortion();
        if(i == 0) groupIdFirst = currId;
        else groupId |= currId;
    }
    if(groupIdFirst != groupId)
        throw MRayError("PrimitiveIds of a surface must "
                        "be the same type!");

    uint32_t sId = surfaceCounter.fetch_add(1);
    surfaces.emplace_back(SurfaceId(sId), std::move(p));
    return SurfaceId(sId);
}

LightSurfaceId TracerBase::CreateLightSurface(LightSurfaceParams p)
{
    uint32_t lightSId = lightSurfaceCounter.fetch_add(1);
    lightSurfaces.emplace_back(LightSurfaceId(lightSId), std::move(p));
    return LightSurfaceId(lightSId);
}

CamSurfaceId TracerBase::CreateCameraSurface(CameraSurfaceParams p)
{
    uint32_t camSId = camSurfaceCounter.fetch_add(1);
    cameraSurfaces.emplace_back(CamSurfaceId(camSId), std::move(p));
    return CamSurfaceId(camSId);
}

void TracerBase::CommitSurfaces(AcceleratorType type)
{
    // Pack the surfaces via transform / primitive
    //
    // PG TG {S0,...,SN},  -> AccelGroup (One AccelInstance per S)
    // PG TG {S0,...,SN},  -> AccelGroup
    // ....
    // PG TG {S0,...,SN}   -> AccelGroup
    auto& surfList = surfaces.Vec();
    using SurfP = Pair<SurfaceId, SurfaceParams>;
    std::stable_sort(surfList.begin(), surfList.end(),
                     [](const SurfP& left, const SurfP& right) -> bool
    {
        return (left.second.primBatches.front() <
                right.second.primBatches.front());
    });
    //
    std::stable_sort(surfList.begin(), surfList.end(),
                     [](const SurfP& left, const SurfP& right) -> bool
    {
        return (left.second.transformId < right.second.transformId);
    });
    // Do the same thing for lights
    using LightSurfP = Pair<LightSurfaceId, LightSurfaceParams>;
    auto& lSurfList = lightSurfaces.Vec();
    auto lPartitionEnd = std::partition(lSurfList.begin(), lSurfList.end(),
    [this](const LightSurfP& lSurf)
    {
        const auto& map = lightGroups.Map();
        LightKey lKey(static_cast<uint32_t>(lSurf.second.lightId));
        auto gId = LightGroupId(lKey.FetchBatchPortion());
        return map.at(gId)->IsPrimitiveBacked();
    });
    std::stable_sort(lSurfList.begin(), lSurfList.end(),
    [](const LightSurfP& left, const LightSurfP& right)
    {
        return left.second.lightId < right.second.lightId;
    });
    std::stable_sort(lSurfList.begin(), lSurfList.end(),
    [](const LightSurfP& left, const LightSurfP& right)
    {
        return left.second.transformId < right.second.transformId;
    });
    // Send it!
    accelerator = typeGenerators.baseAcceleratorGenerator.at(type)(threadPool, gpuSystem,
                                                                   typeGenerators.accelGeneratorMap.at(type),
                                                                   typeGenerators.accelWorkGeneratorMap.at(type));
    accelerator->Construct(BaseAccelConstructParams
    {
        .texViewMap = texViewMap,
        .primGroups = primGroups.Map(),
        .lightGroups = lightGroups.Map(),
        .transformGroups = transGroups.Map(),
        .mSurfList = surfaces.Vec(),
        .lSurfList = Span<const LightSurfP>(lSurfList.begin(),
                                            lPartitionEnd)
    });
}

RendererId TracerBase::CreateRenderer(std::string typeName)
{
    uint32_t rId = redererCounter.fetch_add(1u);
    auto renderer = typeGenerators.rendererGenerator.at(typeName)(gpuSystem);
    renderers.try_emplace(RendererId(rId), std::move(renderer));
    return RendererId(rId);
}

void TracerBase::DestroyRenderer(RendererId rId)
{
    renderers.remove_at(rId);
}

void TracerBase::CommitRendererReservations(RendererId rId)
{
    renderers.at(rId)->Commit();
}

bool TracerBase::IsRendererCommitted(RendererId rId) const
{
    return renderers.at(rId)->IsInCommitState();
}

void TracerBase::PushRendererAttribute(RendererId rId,
                                       uint32_t attribIndex,
                                       TransientData data)
{
    // TODO: Change this
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    renderers.at(rId)->PushAttribute(attribIndex, std::move(data),
                                     queue);
}

void TracerBase::StartRender(RendererId rId, CamSurfaceId cId,
                             RenderImageParams params)
{
    currentRenderer = renderers.at(rId).get();
    auto camKey = CameraKey(static_cast<uint32_t>(cId));
    currentRenderer->StartRender(params, camKey);
}

void TracerBase::StopRender()
{
    currentRenderer->StopRender();
}

RendererOutput TracerBase::DoRenderWork()
{
    return currentRenderer->DoRender();
}

void TracerBase::ClearAll()
{
    primGroups.clear();
    camGroups.clear();
    mediumGroups.clear();
    matGroups.clear();
    transGroups.clear();
    lightGroups.clear();
    renderers.clear();
    //
    primGroupCounter = 0;
    camGroupCounter = 0;
    mediumGroupCounter = 0;
    matGroupCounter = 0;
    transGroupCounter = 0;
    lightGroupCounter = 0;
    surfaceCounter = 0;
    lightSurfaceCounter = 0;
    camSurfaceCounter = 0;
    textureCounter = 0;
}