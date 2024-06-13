#include "TracerBase.h"
#include <BS/BS_thread_pool.hpp>

// TODO: This is not good, we need to instantiate
// to get the virtual function, change this later
// (change this with what though?)
template <class MapOut, class Map, class... Args>
static void InstantiateAndGetAttribInfo(MapOut& result,
                                        const Map& map,
                                        Args&&... args)
{
    for(const auto& kv : map)
    {
        auto instance = kv.second(std::forward<Args>(args)...);
        result.emplace(kv.first, instance->AttributeInfo());
    }
};

void TracerBase::PopulateAttribInfoAndTypeLists()
{
    auto FlattenMapKeys = [](const auto& map)
    {
        TypeNameList list; list.reserve(map.size());
        for(const auto& kv : map)
            list.emplace_back(kv.first);
        return list;
    };

    primTypes = FlattenMapKeys(typeGenerators.primGenerator);
    camTypes = FlattenMapKeys(typeGenerators.camGenerator);
    medTypes = FlattenMapKeys(typeGenerators.medGenerator);
    matTypes = FlattenMapKeys(typeGenerators.matGenerator);
    transTypes = FlattenMapKeys(typeGenerators.transGenerator);
    lightTypes = FlattenMapKeys(typeGenerators.lightGenerator);
    rendererTypes = FlattenMapKeys(typeGenerators.rendererGenerator);

    //
    InstantiateAndGetAttribInfo(primAttributeInfoMap,
                                typeGenerators.primGenerator,
                                0u, gpuSystem);
    InstantiateAndGetAttribInfo(camAttributeInfoMap,
                                typeGenerators.camGenerator,
                                0u, gpuSystem);
    InstantiateAndGetAttribInfo(medAttributeInfoMap,
                                typeGenerators.medGenerator,
                                0u, gpuSystem, TextureViewMap{});
    InstantiateAndGetAttribInfo(matAttributeInfoMap,
                                typeGenerators.matGenerator,
                                0u, gpuSystem, TextureViewMap{});
    InstantiateAndGetAttribInfo(transAttributeInfoMap,
                                typeGenerators.transGenerator,
                                0u, gpuSystem);
    InstantiateAndGetAttribInfo(rendererAttributeInfoMap,
                                typeGenerators.rendererGenerator,
                                gpuSystem);

    // Light is special so write it by hand
    // Find instantiate the prim again...
    PrimGroupEmpty emptyPG(0u, gpuSystem);
    for(const auto& kv : typeGenerators.lightGenerator)
    {
        using namespace std::string_view_literals;
        LightGroupPtr instance = nullptr;
        PrimGroupPtr pg = nullptr;
        if(kv.first.find("Prim"sv) != std::string_view::npos)
        {
            size_t loc = kv.first.find(TracerConstants::PRIM_PREFIX);
            auto primType = kv.first.substr(loc);
            auto pgGen = typeGenerators.primGenerator.at(primType);
            if(!pgGen)
            {
                throw MRayError("\"{}\" requires prim type \"{}\" but "
                                "tracer does not support this prim type",
                                kv.first, primType);
            }
            // TODO: This is dangling reference..
            // We will not use pg, but this is UB so gg...
            pg = pgGen.value()(0, gpuSystem);
            instance = kv.second(0u, gpuSystem,
                                 TextureViewMap{},
                                 *pg.get());
        }
        else
        {
            instance = kv.second(0u, gpuSystem,
                                 TextureViewMap{},
                                 emptyPG);
        }
        lightAttributeInfoMap.emplace(kv.first, instance->AttributeInfo());
    }
}

TracerBase::TracerBase(const TypeGeneratorPack& tGen,
                       const TracerParameters& tParams)
    : threadPool(nullptr)
    , typeGenerators(tGen)
    , tracerParams(tParams)
    , texMem(gpuSystem)

{}

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

TypeNameList TracerBase::Renderers() const
{
    return rendererTypes;
}

PrimAttributeInfoList TracerBase::AttributeInfo(PrimGroupId id) const
{
    auto val = primGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

CamAttributeInfoList TracerBase::AttributeInfo(CameraGroupId id) const
{
    auto val = camGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

MediumAttributeInfoList TracerBase::AttributeInfo(MediumGroupId id) const
{
    auto val = mediumGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

MatAttributeInfoList TracerBase::AttributeInfo(MatGroupId id) const
{
    auto val = matGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

TransAttributeInfoList TracerBase::AttributeInfo(TransGroupId id) const
{
    auto val = transGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

LightAttributeInfoList TracerBase::AttributeInfo(LightGroupId id) const
{
    auto val = lightGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

RendererAttributeInfoList TracerBase::AttributeInfo(RendererId id) const
{
    auto val = renderers.at(id);
    if(!val)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<uint32_t>(id));
    }
    return val.value().get()->AttributeInfo();
}

PrimAttributeInfoList TracerBase::AttributeInfoPrim(std::string_view name) const
{
    auto pInfo = primAttributeInfoMap.at(name);
    if(!pInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return pInfo.value();
}

CamAttributeInfoList TracerBase::AttributeInfoCam(std::string_view name) const
{
    auto cInfo = camAttributeInfoMap.at(name);
    if(!cInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return cInfo.value();
}

MediumAttributeInfoList TracerBase::AttributeInfoMedium(std::string_view name) const
{
    auto mInfo = medAttributeInfoMap.at(name);
    if(!mInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return mInfo.value();
}

MatAttributeInfoList TracerBase::AttributeInfoMat(std::string_view name) const
{
    auto mtInfo = matAttributeInfoMap.at(name);
    if(!mtInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return mtInfo.value();
}

TransAttributeInfoList TracerBase::AttributeInfoTrans(std::string_view name) const
{
    auto tInfo = transAttributeInfoMap.at(name);
    if(!tInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return tInfo.value();
}

LightAttributeInfoList TracerBase::AttributeInfoLight(std::string_view name) const
{
    auto lInfo = lightAttributeInfoMap.at(name);
    if(!lInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return lInfo.value();
}

RendererAttributeInfoList TracerBase::AttributeInfoRenderer(std::string_view name) const
{
    auto rInfo = rendererAttributeInfoMap.at(name);
    if(!rInfo)
    {
        throw MRayError("Unable to find type \"{}\"", name);
    }
    return rInfo.value();
}

std::string TracerBase::TypeName(PrimGroupId id) const
{
    auto val = primGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(CameraGroupId id) const
{
    auto val = camGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(MediumGroupId id) const
{
    auto val = mediumGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(MatGroupId id) const
{
    auto val = matGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(TransGroupId id) const
{
    auto val = transGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(LightGroupId id) const
{
    auto val = lightGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(RendererId id) const
{
    auto val = renderers.at(id);
    if(!val)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<uint32_t>(id));
    }
    return std::string(val.value().get()->Name());
}

PrimGroupId TracerBase::CreatePrimitiveGroup(std::string name)
{
    auto genFunc = typeGenerators.primGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }

    uint32_t idInt = primGroupCounter.fetch_add(1);
    PrimGroupId id = static_cast<PrimGroupId>(idInt);
    primGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id),
                                       gpuSystem));
    return id;
}

PrimBatchId TracerBase::ReservePrimitiveBatch(PrimGroupId id, PrimCount count)
{
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<AttributeCountList> input = {{count.primCount, count.attributeCount}};
    std::vector<PrimBatchKey> output;
    output = primGroup.value().get()->Reserve(input);
    using T = typename PrimBatchKey::Type;
    return PrimBatchId(static_cast<T>(output.front()));
}

PrimBatchIdList TracerBase::ReservePrimitiveBatches(PrimGroupId id,
                                                    std::vector<PrimCount> primCounts)
{
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<AttributeCountList> input;
    input.reserve(primCounts.size());
    for(const auto& pc : primCounts)
        input.push_back({pc.primCount, pc.attributeCount});

    std::vector<PrimBatchKey> output;
    output = primGroup.value().get()->Reserve(input);

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
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(id));
    }
    primGroup.value().get()->CommitReservations();
}

bool TracerBase::IsPrimCommitted(PrimGroupId id) const
{
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(id));
    }
    return primGroup.value().get()->IsInCommitState();
}

void TracerBase::PushPrimAttribute(PrimGroupId gId,
                                   PrimBatchId batchId,
                                   uint32_t attribIndex,
                                   TransientData data)
{
    auto primGroup = primGroups.at(gId);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(gId));
    }

    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);

    PrimBatchKey key(static_cast<uint32_t>(batchId));
    primGroup.value().get()->PushAttribute(key, attribIndex,
                                           std::move(data),
                                           queue);
}

void TracerBase::PushPrimAttribute(PrimGroupId gId,
                                   PrimBatchId batchId,
                                   uint32_t attribIndex,
                                   Vector2ui subBatchRange,
                                   TransientData data)
{
    auto primGroup = primGroups.at(gId);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<uint32_t>(gId));
    }

    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);

    PrimBatchKey key(static_cast<uint32_t>(batchId));
    primGroup.value().get()->PushAttribute(key, attribIndex,
                                           subBatchRange,
                                           std::move(data),
                                           queue);
}

MatGroupId TracerBase::CreateMaterialGroup(std::string name)
{
    auto genFunc = typeGenerators.matGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    uint32_t idInt = matGroupCounter.fetch_add(1);
    MatGroupId id = static_cast<MatGroupId>(idInt);
    matGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id),
                                              gpuSystem,
                                              texMem.TextureViews()));
    return id;
}

MaterialId TracerBase::ReserveMaterial(MatGroupId id,
                                       AttributeCountList count,
                                       MediumPair mediumPair)
{
    auto matGroup = matGroups.at(id);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    MediumKeyPairList medPairList;
    medPairList.reserve(1);
    using MedKT = typename MediumKey::Type;
    medPairList.emplace_back(MediumKey(static_cast<MedKT>(mediumPair.first)),
                             MediumKey(static_cast<MedKT>(mediumPair.second)));

    std::vector<MaterialKey> output;
    output = matGroup.value().get()->Reserve(attribCountList, medPairList);

    using MatKT = typename MaterialKey::Type;
    MaterialId result = MaterialId(static_cast<MatKT>(output.front()));
    return result;
}

MaterialIdList TracerBase::ReserveMaterials(MatGroupId id,
                                            std::vector<AttributeCountList> countList,
                                            std::vector<MediumPair> medPairs)
{
    auto matGroup = matGroups.at(id);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(id));
    }

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
    output = matGroup.value().get()->Reserve(countList, medPairList);

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
    auto matGroup = matGroups.at(id);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(id));
    }
    matGroup.value().get()->CommitReservations();
}

bool TracerBase::IsMatCommitted(MatGroupId id) const
{
    auto matGroup = matGroups.at(id);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(id));
    }
    return matGroup.value().get()->IsInCommitState();
}

void TracerBase::PushMatAttribute(MatGroupId gId, Vector2ui matRange,
                                  uint32_t attribIndex,
                                  TransientData data)
{
    auto matGroup = matGroups.at(gId);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(gId));
    }
    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MaterialKey::Type;
    auto keyStart = MaterialKey(static_cast<T>(matRange[0]));
    auto keyEnd = MaterialKey(static_cast<T>(matRange[1]));
    matGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(data), queue);
}

void TracerBase::PushMatAttribute(MatGroupId gId, Vector2ui matRange,
                                  uint32_t attribIndex, TransientData data,
                                  std::vector<Optional<TextureId>> textures)
{
    auto matGroup = matGroups.at(gId);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MaterialKey::Type;
    auto keyStart = MaterialKey(static_cast<T>(matRange[0]));
    auto keyEnd = MaterialKey(static_cast<T>(matRange[1]));
    if(data.IsEmpty())
    {
        matGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                                 std::move(textures), queue);
    }
    else
    {
        matGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                                 std::move(data),
                                                 std::move(textures), queue);
    }
}

void TracerBase::PushMatAttribute(MatGroupId gId, Vector2ui matRange,
                                  uint32_t attribIndex,
                                  std::vector<TextureId> textures)
{
    auto matGroup = matGroups.at(gId);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<uint32_t>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MaterialKey::Type;
    auto keyStart = MaterialKey(static_cast<T>(matRange[0]));
    auto keyEnd = MaterialKey(static_cast<T>(matRange[1]));
    matGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                             std::move(textures), queue);
}

TextureId TracerBase::CreateTexture2D(Vector2ui size, uint32_t mipLevel,
                                      MRayTextureParameters p)
{
    return texMem.CreateTexture2D(size, mipLevel, p);
}

TextureId TracerBase::CreateTexture3D(Vector3ui size, uint32_t mipLevel,
                                      MRayTextureParameters p)
{
    return texMem.CreateTexture3D(size, mipLevel, p);
}

void TracerBase::CommitTextures()
{
    texMem.CommitTextures();
}

void TracerBase::PushTextureData(TextureId id, uint32_t mipLevel,
                                 TransientData data)
{
    texMem.PushTextureData(id, mipLevel, std::move(data));
}

TransGroupId TracerBase::CreateTransformGroup(std::string name)
{
    auto genFunc = typeGenerators.transGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    uint32_t idInt = transGroupCounter.fetch_add(1);
    TransGroupId id = static_cast<TransGroupId>(idInt);
    transGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id), gpuSystem));
    return id;
}

TransformId TracerBase::ReserveTransformation(TransGroupId id, AttributeCountList count)
{
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<TransformKey> output;
    output = transGroup.value().get()->Reserve(attribCountList);

    using T = typename TransformKey::Type;
    TransformId result = TransformId(static_cast<T>(output.front()));
    return result;
}

TransformIdList TracerBase::ReserveTransformations(TransGroupId id,
                                                   std::vector<AttributeCountList> countList)
{
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<TransformKey> output;
    output = transGroup.value().get()->Reserve(countList);

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
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(id));
    }
    transGroup.value().get()->CommitReservations();
}

bool TracerBase::IsTransCommitted(TransGroupId id) const
{
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(id));
    }
    return transGroup.value().get()->IsInCommitState();
}

void TracerBase::PushTransAttribute(TransGroupId gId, Vector2ui transRange,
                                    uint32_t attribIndex,
                                    TransientData data)
{
    auto transGroup = transGroups.at(gId);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<uint32_t>(gId));
    }

    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename TransformKey::Type;
    auto keyStart = TransformKey(static_cast<T>(transRange[0]));
    auto keyEnd = TransformKey(static_cast<T>(transRange[1]));
    transGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                            std::move(data), queue);
}

LightGroupId TracerBase::CreateLightGroup(std::string name,
                                          PrimGroupId primGId)
{
    auto genFunc = typeGenerators.lightGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    auto primGroup = primGroups.at(primGId);
    if(!primGroup)
    {
        throw MRayError("Unable to locate PrimitiveGroup({})",
                        static_cast<uint32_t>(primGId));
    }
    uint32_t idInt = lightGroupCounter.fetch_add(1);
    LightGroupId id = static_cast<LightGroupId>(idInt);
    GenericGroupPrimitiveT& primGroupPtr = *primGroup.value().get().get();
    lightGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id), gpuSystem,
                                                texMem.TextureViews(),
                                                primGroupPtr));
    return id;
}

LightId TracerBase::ReserveLight(LightGroupId id,
                                 AttributeCountList count,
                                 PrimBatchId primId)
{
    auto lightGroup = lightGroups.at(id);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    PrimBatchList primList;
    primList.reserve(1);
    using PrimBatchKT = typename PrimBatchKey::Type;
    primList.emplace_back(PrimBatchKey(static_cast<PrimBatchKT>(primId)));

    std::vector<LightKey> output;
    output = lightGroup.value().get()->Reserve(attribCountList, primList);

    using LightKT = typename LightKey::Type;
    LightId result = LightId(static_cast<LightKT>(output.front()));
    return result;
}

LightIdList TracerBase::ReserveLights(LightGroupId id,
                                      std::vector<AttributeCountList> countList,
                                      std::vector<PrimBatchId> primBatches)
{
    auto lightGroupOpt = lightGroups.at(id);
    if(!lightGroupOpt)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(id));
    }
    const LightGroupPtr& lightGroup = lightGroupOpt.value().get();

    PrimBatchList primList;
    if(lightGroup->IsPrimitiveBacked())
    {
        assert(primBatches.size() == countList.size());

        primList.reserve(countList.size());
        for(size_t i = 0; i < countList.size(); i++)
        {
            using PrimBatchKT = typename PrimBatchKey::Type;
            primList.emplace_back(PrimBatchKey(static_cast<PrimBatchKT>(primBatches[i])));
        }
    }

    std::vector<LightKey> output;
    output = lightGroup->Reserve(countList, primList);

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
    auto lightGroup = lightGroups.at(id);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(id));
    }
    lightGroup.value().get()->CommitReservations();
}

bool TracerBase::IsLightCommitted(LightGroupId id) const
{
    auto lightGroup = lightGroups.at(id);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(id));
    }

    return lightGroup.value().get()->IsInCommitState();
}

void TracerBase::PushLightAttribute(LightGroupId gId, Vector2ui lightRange,
                                    uint32_t attribIndex,
                                    TransientData data)
{
    auto lightGroup = lightGroups.at(gId);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename LightKey::Type;
    auto keyStart = LightKey(static_cast<T>(lightRange[0]));
    auto keyEnd = LightKey(static_cast<T>(lightRange[1]));
    lightGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                            std::move(data), queue);
}

void TracerBase::PushLightAttribute(LightGroupId gId, Vector2ui lightRange,
                                    uint32_t attribIndex,
                                    TransientData data,
                                    std::vector<Optional<TextureId>> textures)
{
    auto lightGroup = lightGroups.at(gId);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename LightKey::Type;
    auto keyStart = LightKey(static_cast<T>(lightRange[0]));
    auto keyEnd = LightKey(static_cast<T>(lightRange[1]));

    if(data.IsEmpty())
    {
        lightGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                                   std::move(textures), queue);
    }
    else
    {

        lightGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                                   std::move(data), std::move(textures),
                                                   queue);
    }
}

void TracerBase::PushLightAttribute(LightGroupId gId, Vector2ui lightRange,
                                    uint32_t attribIndex,
                                    std::vector<TextureId> textures)
{
    auto lightGroup = lightGroups.at(gId);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<uint32_t>(gId));
    }

    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename LightKey::Type;
    auto keyStart = LightKey(static_cast<T>(lightRange[0]));
    auto keyEnd = LightKey(static_cast<T>(lightRange[1]));
    lightGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                               std::move(textures), queue);
}

CameraGroupId TracerBase::CreateCameraGroup(std::string name)
{
    auto genFunc = typeGenerators.camGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    uint32_t idInt = camGroupCounter.fetch_add(1);
    CameraGroupId id = static_cast<CameraGroupId>(idInt);
    camGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id), gpuSystem));
    return id;
}

CameraId TracerBase::ReserveCamera(CameraGroupId id,
                                   AttributeCountList count)
{
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<CameraKey> output;
    output = camGroup.value().get()->Reserve(attribCountList);

    using T = typename CameraKey::Type;
    CameraId result = CameraId(static_cast<T>(output.front()));
    return result;
}

CameraIdList TracerBase::ReserveCameras(CameraGroupId id,
                                        std::vector<AttributeCountList> countList)
{
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(id));
    }
    std::vector<CameraKey> output;
    output = camGroup.value().get()->Reserve(countList);

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
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(id));
    }
    camGroup.value().get()->CommitReservations();
}

bool TracerBase::IsCamCommitted(CameraGroupId id) const
{
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(id));
    }
    return camGroup.value().get()->IsInCommitState();
}

void TracerBase::PushCamAttribute(CameraGroupId gId, Vector2ui camRange,
                                  uint32_t attribIndex,
                                  TransientData data)
{
    auto camGroup = camGroups.at(gId);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename CameraKey::Type;
    auto keyStart = CameraKey(static_cast<T>(camRange[0]));
    auto keyEnd = CameraKey(static_cast<T>(camRange[1]));
    camGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(data), queue);
}

MediumGroupId TracerBase::CreateMediumGroup(std::string name)
{
    auto genFunc = typeGenerators.medGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    uint32_t idInt = mediumGroupCounter.fetch_add(1);
    MediumGroupId id = static_cast<MediumGroupId>(idInt);
    mediumGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id),
                                                 gpuSystem,
                                                 texMem.TextureViews()));
    return id;
}

MediumId TracerBase::ReserveMedium(MediumGroupId id, AttributeCountList count)
{
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(id));
    }
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<MediumKey> output;
    output = medGroup.value().get()->Reserve(attribCountList);

    using T = typename MediumKey::Type;
    MediumId result = MediumId(static_cast<T>(output.front()));
    return result;
}

MediumIdList TracerBase::ReserveMediums(MediumGroupId id,
                                        std::vector<AttributeCountList> countList)
{
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(id));
    }
    std::vector<MediumKey> output;
    output = medGroup.value().get()->Reserve(countList);
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
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(id));
    }
    medGroup.value().get()->CommitReservations();
}

bool TracerBase::IsMediumCommitted(MediumGroupId id) const
{
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(id));
    }
    return medGroup.value().get()->IsInCommitState();
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, Vector2ui mediumRange,
                                     uint32_t attribIndex,
                                     TransientData data)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MediumKey::Type;
    auto keyStart = MediumKey(static_cast<T>(mediumRange[0]));
    auto keyEnd = MediumKey(static_cast<T>(mediumRange[1]));
    auto medGroup = mediumGroups.at(gId);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(gId));
    }
    medGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
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
    auto medGroup = mediumGroups.at(gId);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(gId));
    }
    if(data.IsEmpty())
    {
        medGroup.value().get()->PushTexAttribute(keyStart, keyEnd,
                                                 attribIndex,
                                                 std::move(textures),
                                                 queue);
    }
    else
    {
        medGroup.value().get()->PushTexAttribute(keyStart, keyEnd,
                                                 attribIndex,
                                                 std::move(data),
                                                 std::move(textures),
                                                 queue);
    }
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, Vector2ui mediumRange,
                                     uint32_t attribIndex,
                                     std::vector<TextureId> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    using T = typename MediumKey::Type;
    auto keyStart = MediumKey(static_cast<T>(mediumRange[0]));
    auto keyEnd = MediumKey(static_cast<T>(mediumRange[1]));
    auto medGroup = mediumGroups.at(gId);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<uint32_t>(gId));
    }
    medGroup.value().get()->PushTexAttribute(keyStart, keyEnd, attribIndex,
                                             std::move(textures), queue);
}

SurfaceId TracerBase::CreateSurface(SurfaceParams p)
{
    // Validate that all batches are in the same group
    auto loc = std::adjacent_find(p.primBatches.cbegin(),
                                  p.primBatches.cend(),
    [](PrimBatchId lhs, PrimBatchId rhs)
    {
        PrimBatchKey keyL(static_cast<uint32_t>(lhs));
        PrimBatchKey keyR(static_cast<uint32_t>(rhs));
        return keyL.FetchBatchPortion() != keyR.FetchBatchPortion();
    });

    if(loc != p.primBatches.cend())
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

SurfaceCommitResult TracerBase::CommitSurfaces()
{
    // Pack the surfaces via transform / primitive
    //
    // PG TG {S0,...,SN},  -> AccelGroup (One AccelInstance per S)
    // PG TG {S0,...,SN},  -> AccelGroup
    // ....
    // PG TG {S0,...,SN}   -> AccelGroup
    auto& surfList = surfaces.Vec();
    using SurfP = Pair<SurfaceId, SurfaceParams>;
    std::sort(surfList.begin(), surfList.end(),
              [](const SurfP& left, const SurfP& right) -> bool
    {
        return (Tuple(left.second.primBatches.front(), left.second.transformId) <
                Tuple(right.second.primBatches.front(), right.second.transformId));
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
        return map.at(gId).value().get()->IsPrimitiveBacked();
    });
    std::sort(lSurfList.begin(), lSurfList.end(),
    [](const LightSurfP& left, const LightSurfP& right)
    {
        return (Tuple(left.second.lightId, left.second.transformId) <
                Tuple(right.second.lightId, right.second.transformId));
    });

    // Send it!
    AcceleratorType type = tracerParams.accelMode;
    auto accelGenerator = typeGenerators.baseAcceleratorGenerator.at(type);
    auto accelGGeneratorMap = typeGenerators.accelGeneratorMap.at(type);
    auto accelWGeneratorMap = typeGenerators.accelWorkGeneratorMap.at(type);
    if(!accelGenerator ||
       !accelGGeneratorMap ||
       !accelWGeneratorMap)
    {
        throw MRayError("Unable to find accelerator generators for type \"{}\"",
                        type);
    }
    accelerator = accelGenerator.value().get()(*threadPool, gpuSystem,
                                 accelGGeneratorMap.value().get(),
                                 accelWGeneratorMap.value().get());
    accelerator->Construct(BaseAccelConstructParams
    {
        .texViewMap = texMem.TextureViews(),
        .primGroups = primGroups.Map(),
        .lightGroups = lightGroups.Map(),
        .transformGroups = transGroups.Map(),
        .mSurfList = surfaces.Vec(),
        .lSurfList = Span<const LightSurfP>(lSurfList.begin(),
                                            lPartitionEnd)
    });

    // Synchronize All here, we may time this
    gpuSystem.SyncAll();
    return SurfaceCommitResult
    {
        .aabb = accelerator->SceneAABB(),
        .instanceCount = accelerator->TotalInstanceCount(),
        .acceleratorCount = accelerator->TotalAccelCount()
    };
}

CameraTransform TracerBase::GetCamTransform(CamSurfaceId camSurfId) const
{
    // Unfortunately these are vector we do linear search
    // But there maybe like 5-10 cams on a scene maybe so its ok
    auto loc = std::find_if(cameraSurfaces.Vec().cbegin(), cameraSurfaces.Vec().cend(),
    [camSurfId](const Pair<CamSurfaceId, CameraSurfaceParams>& p)
    {
        return camSurfId == p.first;
    });
    if(loc == cameraSurfaces.Vec().cend())
        throw MRayError("Unable to find Camera Surface ({})",
                        static_cast<uint32_t>(camSurfId));

    CameraId id = loc->second.cameraId;
    using T = typename CameraKey::Type;
    auto key = CameraKey(static_cast<T>(id));
    auto gId = CameraGroupId(key.FetchBatchPortion());
    auto camGroup = camGroups.at(gId); ;
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<uint32_t>(gId));
    }
    return camGroup.value().get()->AcquireCameraTransform(key);
}

RendererId TracerBase::CreateRenderer(std::string typeName)
{
    auto rendererGen = typeGenerators.rendererGenerator.at(typeName);
    if(!rendererGen)
    {
        throw MRayError("Unable to find generator for {}",
                        typeName);
    }
    auto renderer = rendererGen.value()(gpuSystem);
    uint32_t rId = redererCounter.fetch_add(1u);
    renderers.try_emplace(RendererId(rId), std::move(renderer));
    return RendererId(rId);
}

void TracerBase::DestroyRenderer(RendererId rId)
{
    renderers.remove_at(rId);
}

void TracerBase::CommitRendererReservations(RendererId rId)
{
    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<uint32_t>(rId));
    }
    renderer.value().get()->Commit();
}

bool TracerBase::IsRendererCommitted(RendererId rId) const
{
    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<uint32_t>(rId));
    }
    return renderer.value().get()->IsInCommitState();
}

void TracerBase::PushRendererAttribute(RendererId rId,
                                       uint32_t attribIndex,
                                       TransientData data)
{
    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<uint32_t>(rId));
    }
    // TODO: Change this
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    renderer.value().get()->PushAttribute(attribIndex, std::move(data),
                                          queue);
}

RenderBufferInfo TracerBase::StartRender(RendererId rId, CamSurfaceId cId,
                                         RenderImageParams params,
                                         Optional<CameraTransform> optionalTransform)
{
    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<uint32_t>(rId));
    }
    currentRenderer = renderer.value().get().get();
    auto camKey = CameraKey(static_cast<uint32_t>(cId));
    return currentRenderer->StartRender(params, camKey);
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

    accelerator.reset(nullptr);
    surfaces.clear();
    lightSurfaces.clear();
    cameraSurfaces.clear();

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

    texMem.Clear();

    // GroupId 0 is reserved for empty primitive
    CreatePrimitiveGroup(std::string(TracerConstants::EmptyPrimName));
}

GPUThreadInitFunction TracerBase::GetThreadInitFunction() const
{
    return gpuSystem.GetThreadInitFunction();
}

void TracerBase::SetThreadPool(BS::thread_pool& tp)
{
    threadPool = &tp;
}

size_t TracerBase::TotalDeviceMemory() const
{
    return gpuSystem.TotalMemory();
}

size_t TracerBase::UsedDeviceMemory() const
{
    auto FetchMemUsage = [](const auto& groupMap)
    {
        size_t mem = 0;
        for(const auto& [_, group] : groupMap)
        {
            mem += group->GPUMemoryUsage();
        }
        return mem;
    };

    size_t totalMem = 0;
    totalMem += FetchMemUsage(primGroups.Map());
    totalMem += FetchMemUsage(camGroups.Map());
    totalMem += FetchMemUsage(mediumGroups.Map());
    totalMem += FetchMemUsage(matGroups.Map());
    totalMem += FetchMemUsage(transGroups.Map());
    totalMem += FetchMemUsage(lightGroups.Map());
    totalMem += FetchMemUsage(renderers.Map());
    if(accelerator)
        totalMem += accelerator->GPUMemoryUsage();
    return totalMem;
}