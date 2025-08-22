#include "TracerBase.h"

#include "Core/Timer.h"

using FilterFuncPair = std::pair< const FilterType::E, TexFilterGenerator>;
template<class T>
static constexpr auto FilterGenFuncPack = FilterFuncPair

{
    T::TypeName,
    &GenerateType<TextureFilterI, T, const GPUSystem&, Float>
};

static constexpr const std::initializer_list<FilterFuncPair> FilterGenFuncList =
{
    FilterGenFuncPack<TextureFilterBox>,
    FilterGenFuncPack<TextureFilterTent>,
    FilterGenFuncPack<TextureFilterGaussian>,
    FilterGenFuncPack<TextureFilterMitchellNetravali>
};

using RNGGenPair = std::pair<const SamplerType::E, RNGGenerator>;

template<class T>
static constexpr auto RNGGenFuncPack = RNGGenPair
{
    T::TypeName,
    &GenerateType<RNGeneratorGroupI, T, uint32_t, uint64_t,
                  const GPUSystem&, ThreadPool&>
};

static constexpr std::initializer_list<RNGGenPair> RNGGenFuncList =
{
    RNGGenFuncPack<RNGGroupIndependent>
};

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
                                0u, gpuSystem, TextureViewMap{}, TextureMap{});
    InstantiateAndGetAttribInfo(matAttributeInfoMap,
                                typeGenerators.matGenerator,
                                0u, gpuSystem, TextureViewMap{}, TextureMap{});
    InstantiateAndGetAttribInfo(transAttributeInfoMap,
                                typeGenerators.transGenerator,
                                0u, gpuSystem);
    // InstantiateAndGetAttribInfo(rendererAttributeInfoMap,
    //                             typeGenerators.rendererGenerator,
    //                             RenderImagePtr(),
    //                             GenerateTracerView(),
    //                             *globalThreadPool,
    //                             gpuSystem,
    //                             RenderWorkPack());

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
                                 TextureMap{},
                                 *pg.get());
        }
        else
        {
            instance = kv.second(0u, gpuSystem,
                                 TextureViewMap{},
                                 TextureMap{},
                                 emptyPG);
        }
        lightAttributeInfoMap.emplace(kv.first, instance->AttributeInfo());
    }
}

TracerView TracerBase::GenerateTracerView()
{
    return TracerView
    {
        .baseAccelerator = *accelerator,
        .primGroups = primGroups.GetMap(),
        .camGroups = camGroups.GetMap(),
        .mediumGroups = mediumGroups.GetMap(),
        .matGroups = matGroups.GetMap(),
        .transGroups = transGroups.GetMap(),
        .lightGroups = lightGroups.GetMap(),
        .textures = texMem.Textures(),
        .textureViews = texMem.TextureViews(),
        .tracerParams = params,
        .filterGenerators = filterGenMap,
        .rngGenerators = rngGenMap,
        .boundarySurface = boundarySurface.second,
        .surfs = surfaces.Vec(),
        .lightSurfs = lightSurfaces.Vec(),
        .camSurfs = cameraSurfaces.Vec(),
        .flattenedSurfaces = flattenedSurfaces
    };
}

void TracerBase::GenerateDefaultGroups()
{

    //PrimGroupId emptyPGId = CreatePrimitiveGroup(std::string(TracerConstants::EmptyPrimName));
    auto genFuncP = typeGenerators.primGenerator.at(TracerConstants::EmptyPrimName);
    uint32_t idInt = primGroupCounter.fetch_add(1);
    PrimGroupId emptyPGId = static_cast<PrimGroupId>(idInt);
    auto primGLoc = primGroups.try_emplace(emptyPGId,
                                           genFuncP.value()(std::move(idInt), gpuSystem));
    assert(emptyPGId == TracerConstants::EmptyPrimGroupId);
    //
    auto genFuncL = typeGenerators.lightGenerator.at(TracerConstants::NullLightName);
    idInt = lightGroupCounter.fetch_add(1);
    LightGroupId nullLGId = static_cast<LightGroupId>(idInt);
    lightGroups.try_emplace(nullLGId,
                            genFuncL.value()(std::move(idInt), gpuSystem,
                                             texMem.TextureViews(),
                                             texMem.Textures(),
                                             *primGLoc.first->second));
    assert(nullLGId == TracerConstants::NullLightGroupId);
    //
    auto genFuncT = typeGenerators.transGenerator.at(TracerConstants::IdentityTransName);
    idInt = transGroupCounter.fetch_add(1);
    TransGroupId identTransId = static_cast<TransGroupId>(idInt);
    transGroups.try_emplace(identTransId, genFuncT.value()(std::move(idInt), gpuSystem));
    assert(identTransId == TracerConstants::IdentityTransGroupId);
    //
    auto genFuncMd = typeGenerators.medGenerator.at(TracerConstants::VacuumMediumName);
    idInt = mediumGroupCounter.fetch_add(1);
    MediumGroupId vacuumMedId = static_cast<MediumGroupId>(idInt);
    mediumGroups.try_emplace(vacuumMedId, genFuncMd.value()(std::move(idInt), gpuSystem,
                                                            texMem.TextureViews(),
                                                            texMem.Textures()));
    assert(vacuumMedId == TracerConstants::VacuumMediumGroupId);
}

TracerBase::TracerBase(const TypeGeneratorPack& tGen,
                       const TracerParameters& tParams)
    : globalThreadPool(nullptr)
    , typeGenerators(tGen)
    , filterGenMap(FilterGenFuncList)
    , rngGenMap(RNGGenFuncList)
    , params(tParams)
    , texMem(gpuSystem, params, filterGenMap)
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
                        static_cast<CommonKey>(id));
    }
    return val.value().get()->AttributeInfo();
}

CamAttributeInfoList TracerBase::AttributeInfo(CameraGroupId id) const
{
    auto val = camGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(id));
    }
    return val.value().get()->AttributeInfo();
}

MediumAttributeInfoList TracerBase::AttributeInfo(MediumGroupId id) const
{
    auto val = mediumGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(id));
    }
    return val.value().get()->AttributeInfo();
}

MatAttributeInfoList TracerBase::AttributeInfo(MatGroupId id) const
{
    auto val = matGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(id));
    }
    return val.value().get()->AttributeInfo();
}

TransAttributeInfoList TracerBase::AttributeInfo(TransGroupId id) const
{
    auto val = transGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(id));
    }
    return val.value().get()->AttributeInfo();
}

LightAttributeInfoList TracerBase::AttributeInfo(LightGroupId id) const
{
    auto val = lightGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(id));
    }
    return val.value().get()->AttributeInfo();
}

RendererAttributeInfoList TracerBase::AttributeInfo(RendererId id) const
{
    auto val = renderers.at(id);
    if(!val)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<CommonKey>(id));
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
    //auto rInfo = rendererAttributeInfoMap.at(name);
    auto rInfo = typeGenerators.rendererAttribMap.at(name);
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
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(CameraGroupId id) const
{
    auto val = camGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(MediumGroupId id) const
{
    auto val = mediumGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(MatGroupId id) const
{
    auto val = matGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(TransGroupId id) const
{
    auto val = transGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(LightGroupId id) const
{
    auto val = lightGroups.at(id);
    if(!val)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

std::string TracerBase::TypeName(RendererId id) const
{
    auto val = renderers.at(id);
    if(!val)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<CommonKey>(id));
    }
    return std::string(val.value().get()->Name());
}

PrimGroupId TracerBase::CreatePrimitiveGroup(std::string name)
{
    if(name == TracerConstants::EmptyPrimName)
        return TracerConstants::EmptyPrimGroupId;

    auto genFunc = typeGenerators.primGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }

    uint32_t idInt = primGroupCounter.fetch_add(1);
    if(idInt > PrimitiveKey::BatchMask)
        throw MRayError("Too many Transform Groups");
    PrimGroupId id = static_cast<PrimGroupId>(idInt);
    primGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id),
                                       gpuSystem));
    return id;
}

PrimBatchId TracerBase::ReservePrimitiveBatch(PrimGroupId id, PrimCount count)
{
    if(id == TracerConstants::EmptyPrimGroupId)
        return TracerConstants::EmptyPrimBatchId;

    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<CommonKey>(id));
    }

    std::vector<AttributeCountList> input = {{count.primCount, count.attributeCount}};
    std::vector<PrimBatchKey> output;
    output = primGroup.value().get()->Reserve(input);
    return std::bit_cast<PrimBatchId>(output.front());
}

PrimBatchIdList TracerBase::ReservePrimitiveBatches(PrimGroupId id,
                                                    std::vector<PrimCount> primCounts)
{
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<CommonKey>(id));
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
        result.push_back(std::bit_cast<PrimBatchId>(key));
    }
    return result;
}

void TracerBase::CommitPrimReservations(PrimGroupId id)
{
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<CommonKey>(id));
    }
    primGroup.value().get()->CommitReservations();
}

bool TracerBase::IsPrimCommitted(PrimGroupId id) const
{
    auto primGroup = primGroups.at(id);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<CommonKey>(id));
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
                        static_cast<CommonKey>(gId));
    }

    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);

    PrimBatchKey key = std::bit_cast<PrimBatchKey>(batchId);
    primGroup.value().get()->PushAttribute(key, attribIndex,
                                           std::move(data),
                                           queue);
}


void TracerBase::TransformPrimitives(PrimGroupId gId,
                                     std::vector<PrimBatchId> primBatchIds,
                                     std::vector<Matrix4x4> transforms)
{
    auto primGroup = primGroups.at(gId);
    if(!primGroup)
    {
        throw MRayError("Unable to find PrimitiveGroup({})",
                        static_cast<CommonKey>(gId));
    }

    // "PrimBatchId" is same as "PrimBatchKey",
    // but it is wrapped around std::vector
    // so we can't bit cast the std::vector
    // copy...
    std::vector<PrimBatchKey> primBatches;
    primBatches.reserve(primBatchIds.size());
    for(auto id : primBatchIds)
        primBatches.push_back(std::bit_cast<PrimBatchKey>(id));

    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    primGroup.value().get()->ApplyTransformations(primBatches,
                                                  transforms, queue);
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
                        static_cast<CommonKey>(gId));
    }

    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);

    PrimBatchKey key = std::bit_cast<PrimBatchKey>(batchId);
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
    if(idInt > MaterialKey::BatchMask)
        throw MRayError("Too many Material Groups");
    matGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id),
                                              gpuSystem,
                                              texMem.TextureViews(),
                                              texMem.Textures()));
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
                        static_cast<CommonKey>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    MediumKeyPairList medPairList;
    medPairList.reserve(1);
    medPairList.emplace_back(std::bit_cast<MediumKey>(mediumPair.first),
                             std::bit_cast<MediumKey>(mediumPair.second));

    std::vector<MaterialKey> output;
    output = matGroup.value().get()->Reserve(attribCountList, medPairList);
    MaterialId result = std::bit_cast<MaterialId>(output.front());
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
                        static_cast<CommonKey>(id));
    }

    MediumKeyPairList medPairList;
    medPairList.reserve(countList.size());
    if(medPairs.empty())
    {
        using TracerConstants::VacuumMediumId;
        MediumKey mKey = std::bit_cast<MediumKey>(VacuumMediumId);
        medPairList.resize(countList.size(), MediumKeyPair{mKey, mKey});
    }
    else for(size_t i = 0; i < countList.size(); i++)
    {
        assert(medPairs.size() == countList.size());
        medPairList.emplace_back(std::bit_cast<MediumKey>(medPairs[i].first),
                                 std::bit_cast<MediumKey>(medPairs[i].second));
    }

    std::vector<MaterialKey> output;
    output = matGroup.value().get()->Reserve(countList, medPairList);

    MaterialIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        result.push_back(std::bit_cast<MaterialId>(key));
    }
    return result;
}

void TracerBase::CommitMatReservations(MatGroupId id)
{
    auto matGroup = matGroups.at(id);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(id));
    }
    matGroup.value().get()->CommitReservations();
}

bool TracerBase::IsMatCommitted(MatGroupId id) const
{
    auto matGroup = matGroups.at(id);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(id));
    }
    return matGroup.value().get()->IsInCommitState();
}

void TracerBase::PushMatAttribute(MatGroupId gId, CommonIdRange matRange,
                                  uint32_t attribIndex,
                                  TransientData data)
{
    auto matGroup = matGroups.at(gId);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(gId));
    }
    // TODO: Change this to utilize muti-gpu/queue
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<MaterialKey>(matRange[0]);
    auto keyEnd = std::bit_cast<MaterialKey>(matRange[1]);
    matGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(data), queue);
}

void TracerBase::PushMatAttribute(MatGroupId gId, CommonIdRange matRange,
                                  uint32_t attribIndex, TransientData data,
                                  std::vector<Optional<TextureId>> textures)
{
    auto matGroup = matGroups.at(gId);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<MaterialKey>(matRange[0]);
    auto keyEnd = std::bit_cast<MaterialKey>(matRange[1]);
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

void TracerBase::PushMatAttribute(MatGroupId gId, CommonIdRange matRange,
                                  uint32_t attribIndex,
                                  std::vector<TextureId> textures)
{
    auto matGroup = matGroups.at(gId);
    if(!matGroup)
    {
        throw MRayError("Unable to find MaterialGroup({})",
                        static_cast<CommonKey>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<MaterialKey>(matRange[0]);
    auto keyEnd = std::bit_cast<MaterialKey>(matRange[1]);
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
    if(name == TracerConstants::IdentityTransName)
        return TracerConstants::IdentityTransGroupId;

    auto genFunc = typeGenerators.transGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    uint32_t idInt = transGroupCounter.fetch_add(1);
    if(idInt > TransformKey::BatchMask)
        throw MRayError("Too many Transform Groups");
    TransGroupId id = static_cast<TransGroupId>(idInt);
    transGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id), gpuSystem));
    return id;
}

TransformId TracerBase::ReserveTransformation(TransGroupId id, AttributeCountList count)
{
    if(id == TracerConstants::IdentityTransGroupId)
        return TracerConstants::IdentityTransformId;

    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<TransformKey> output;
    output = transGroup.value().get()->Reserve(attribCountList);
    TransformId result = std::bit_cast<TransformId>(output.front());
    return result;
}

TransformIdList TracerBase::ReserveTransformations(TransGroupId id,
                                                   std::vector<AttributeCountList> countList)
{
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(id));
    }

    std::vector<TransformKey> output;
    output = transGroup.value().get()->Reserve(countList);

    TransformIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        result.push_back(std::bit_cast<TransformId>(key));
    }
    return result;
}

void TracerBase::CommitTransReservations(TransGroupId id)
{
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(id));
    }
    transGroup.value().get()->CommitReservations();
}

bool TracerBase::IsTransCommitted(TransGroupId id) const
{
    auto transGroup = transGroups.at(id);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(id));
    }
    return transGroup.value().get()->IsInCommitState();
}

void TracerBase::PushTransAttribute(TransGroupId gId, CommonIdRange transRange,
                                    uint32_t attribIndex,
                                    TransientData data)
{
    auto transGroup = transGroups.at(gId);
    if(!transGroup)
    {
        throw MRayError("Unable to find TransformGroup({})",
                        static_cast<CommonKey>(gId));
    }

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<TransformKey>(transRange[0]);
    auto keyEnd = std::bit_cast<TransformKey>(transRange[1]);
    transGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                            std::move(data), queue);
}

LightGroupId TracerBase::CreateLightGroup(std::string name,
                                          PrimGroupId primGId)
{
    if(name == TracerConstants::NullLightName)
        return TracerConstants::NullLightGroupId;

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
                        static_cast<CommonKey>(primGId));
    }
    uint32_t idInt = lightGroupCounter.fetch_add(1);
    if(idInt > LightKey::BatchMask)
        throw MRayError("Too many Light Groups");

    LightGroupId id = static_cast<LightGroupId>(idInt);
    GenericGroupPrimitiveT& primGroupPtr = *primGroup.value().get().get();
    lightGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id), gpuSystem,
                                                texMem.TextureViews(),
                                                texMem.Textures(),
                                                primGroupPtr));
    return id;
}

LightId TracerBase::ReserveLight(LightGroupId id,
                                 AttributeCountList count,
                                 PrimBatchId primId)
{
    if(id == TracerConstants::NullLightGroupId)
        return TracerConstants::NullLightId;

    auto lightGroup = lightGroups.at(id);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    PrimBatchList primList;
    primList.reserve(1);
    primList.emplace_back(std::bit_cast<PrimBatchKey>(primId));

    std::vector<LightKey> output;
    output = lightGroup.value().get()->Reserve(attribCountList, primList);
    LightId result = std::bit_cast<LightId>(output.front());
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
                        static_cast<CommonKey>(id));
    }
    const LightGroupPtr& lightGroup = lightGroupOpt.value().get();

    PrimBatchList primList;
    if(lightGroup->IsPrimitiveBacked())
    {
        assert(primBatches.size() == countList.size());

        primList.reserve(countList.size());
        for(size_t i = 0; i < countList.size(); i++)
        {
            primList.emplace_back(std::bit_cast<PrimBatchKey>(primBatches[i]));
        }
    }

    std::vector<LightKey> output;
    output = lightGroup->Reserve(countList, primList);

    LightIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        result.push_back(std::bit_cast<LightId>(key));
    }
    return result;
}

void TracerBase::CommitLightReservations(LightGroupId id)
{
    auto lightGroup = lightGroups.at(id);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(id));
    }
    lightGroup.value().get()->CommitReservations();
}

bool TracerBase::IsLightCommitted(LightGroupId id) const
{
    auto lightGroup = lightGroups.at(id);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(id));
    }

    return lightGroup.value().get()->IsInCommitState();
}

void TracerBase::PushLightAttribute(LightGroupId gId, CommonIdRange lightRange,
                                    uint32_t attribIndex,
                                    TransientData data)
{
    auto lightGroup = lightGroups.at(gId);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<LightKey>(lightRange[0]);
    auto keyEnd = std::bit_cast<LightKey>(lightRange[1]);
    lightGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                            std::move(data), queue);
}

void TracerBase::PushLightAttribute(LightGroupId gId, CommonIdRange lightRange,
                                    uint32_t attribIndex,
                                    TransientData data,
                                    std::vector<Optional<TextureId>> textures)
{
    auto lightGroup = lightGroups.at(gId);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<LightKey>(lightRange[0]);
    auto keyEnd = std::bit_cast<LightKey>(lightRange[1]);

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

void TracerBase::PushLightAttribute(LightGroupId gId, CommonIdRange lightRange,
                                    uint32_t attribIndex,
                                    std::vector<TextureId> textures)
{
    auto lightGroup = lightGroups.at(gId);
    if(!lightGroup)
    {
        throw MRayError("Unable to find LightGroup({})",
                        static_cast<CommonKey>(gId));
    }

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<LightKey>(lightRange[0]);
    auto keyEnd = std::bit_cast<LightKey>(lightRange[1]);
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
    if(idInt > CameraKey::BatchMask)
        throw MRayError("Too many Camera Groups");
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
                        static_cast<CommonKey>(id));
    }

    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<CameraKey> output;
    output = camGroup.value().get()->Reserve(attribCountList);
    CameraId result = std::bit_cast<CameraId>(output.front());
    return result;
}

CameraIdList TracerBase::ReserveCameras(CameraGroupId id,
                                        std::vector<AttributeCountList> countList)
{
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(id));
    }
    std::vector<CameraKey> output;
    output = camGroup.value().get()->Reserve(countList);

    CameraIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        result.push_back(std::bit_cast<CameraId>(key));
    }
    return result;
}

void TracerBase::CommitCamReservations(CameraGroupId id)
{
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(id));
    }
    camGroup.value().get()->CommitReservations();
}

bool TracerBase::IsCamCommitted(CameraGroupId id) const
{
    auto camGroup = camGroups.at(id);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(id));
    }
    return camGroup.value().get()->IsInCommitState();
}

void TracerBase::PushCamAttribute(CameraGroupId gId, CommonIdRange camRange,
                                  uint32_t attribIndex,
                                  TransientData data)
{
    auto camGroup = camGroups.at(gId);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(gId));
    }
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<CameraKey>(camRange[0]);
    auto keyEnd = std::bit_cast<CameraKey>(camRange[1]);
    camGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(data), queue);
}

MediumGroupId TracerBase::CreateMediumGroup(std::string name)
{
    if(name == TracerConstants::VacuumMediumName)
        return TracerConstants::VacuumMediumGroupId;

    auto genFunc = typeGenerators.medGenerator.at(name);
    if(!genFunc)
    {
        throw MRayError("Unable to find generator for {}",
                        name);
    }
    uint32_t idInt = mediumGroupCounter.fetch_add(1);
    if(idInt > MediumKey::BatchMask)
        throw MRayError("Too many Medium Groups");
    MediumGroupId id = static_cast<MediumGroupId>(idInt);
    mediumGroups.try_emplace(id, genFunc.value()(static_cast<uint32_t>(id),
                                                 gpuSystem,
                                                 texMem.TextureViews(),
                                                 texMem.Textures()));
    return id;
}

MediumId TracerBase::ReserveMedium(MediumGroupId id, AttributeCountList count)
{
    if(id == TracerConstants::VacuumMediumGroupId)
        return TracerConstants::VacuumMediumId;

    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(id));
    }
    std::vector<AttributeCountList> attribCountList;
    attribCountList.reserve(1);
    attribCountList.push_back(count);

    std::vector<MediumKey> output;
    output = medGroup.value().get()->Reserve(attribCountList);
    MediumId result = std::bit_cast<MediumId>(output.front());
    return result;
}

MediumIdList TracerBase::ReserveMediums(MediumGroupId id,
                                        std::vector<AttributeCountList> countList)
{
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(id));
    }
    std::vector<MediumKey> output;
    output = medGroup.value().get()->Reserve(countList);
    MediumIdList result;
    result.reserve(output.size());
    for(const auto& key : output)
    {
        result.push_back(std::bit_cast<MediumId>(key));
    }
    return result;
}

void TracerBase::CommitMediumReservations(MediumGroupId id)
{
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(id));
    }
    medGroup.value().get()->CommitReservations();
}

bool TracerBase::IsMediumCommitted(MediumGroupId id) const
{
    auto medGroup = mediumGroups.at(id);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(id));
    }
    return medGroup.value().get()->IsInCommitState();
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, CommonIdRange mediumRange,
                                     uint32_t attribIndex,
                                     TransientData data)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<MediumKey>(mediumRange[0]);
    auto keyEnd = std::bit_cast<MediumKey>(mediumRange[1]);
    auto medGroup = mediumGroups.at(gId);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(gId));
    }
    medGroup.value().get()->PushAttribute(keyStart, keyEnd, attribIndex,
                                          std::move(data), queue);
}

void TracerBase::PushMediumAttribute(MediumGroupId gId, CommonIdRange mediumRange,
                                     uint32_t attribIndex,
                                     TransientData data,
                                     std::vector<Optional<TextureId>> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<MediumKey>(mediumRange[0]);
    auto keyEnd = std::bit_cast<MediumKey>(mediumRange[1]);
    auto medGroup = mediumGroups.at(gId);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(gId));
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

void TracerBase::PushMediumAttribute(MediumGroupId gId, CommonIdRange mediumRange,
                                     uint32_t attribIndex,
                                     std::vector<TextureId> textures)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    auto keyStart = std::bit_cast<MediumKey>(mediumRange[0]);
    auto keyEnd = std::bit_cast<MediumKey>(mediumRange[1]);
    auto medGroup = mediumGroups.at(gId);
    if(!medGroup)
    {
        throw MRayError("Unable to find MediumGroup({})",
                        static_cast<CommonKey>(gId));
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
        PrimBatchKey keyL = std::bit_cast<PrimBatchKey>(lhs);
        PrimBatchKey keyR = std::bit_cast<PrimBatchKey>(rhs);
        return keyL.FetchBatchPortion() != keyR.FetchBatchPortion();
    });

    if(loc != p.primBatches.cend())
        throw MRayError("PrimitiveIds of a surface must "
                        "be the same type!");

    uint32_t sId = surfaceCounter.fetch_add(1);
    surfaces.emplace_back(SurfaceId(sId), std::move(p));
    return SurfaceId(sId);
}

LightSurfaceId TracerBase::SetBoundarySurface(LightSurfaceParams p)
{
    LightSurfaceId lightSId;
    if(boundarySurface.first == TracerIdInvalid<LightSurfaceId>)
        lightSId = LightSurfaceId(lightSurfaceCounter.fetch_add(1));
    else
        lightSId = boundarySurface.first;

    boundarySurface = {lightSId, p};
    return lightSId;
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
    // Synchronize all here, probably scene load is issued
    // previously. CPU side is synchronized but async calls to GPU may not
    // have finished yet. Wait before finalizing groups
    gpuSystem.SyncAll();

    // Check the boundary surface
    //
    if(boundarySurface.first == TracerIdInvalid<LightSurfaceId>)
    {
        MRAY_WARNING_LOG("No boundary surface is set! Setting a default one "
                         "(NullLight, IdentityTransform, VacuumMedium).");
        boundarySurface.second = LightSurfaceParams{};
    }

    // Finalize the group operations
    // Some groups require post-processing
    // namely triangles (which adjust the local index values
    // to global one) and transforms (which invert the transforms and store)
    GPUQueueIteratorRoundRobin queueIt(gpuSystem);
    // Finalize the texture operations
    texMem.Finalize();
    for(auto& g : primGroups.GetMap())
    { g.second->Finalize(queueIt.Queue()); queueIt.Next(); }
    for(auto& g : camGroups.GetMap())
    { g.second->Finalize(queueIt.Queue()); queueIt.Next(); }
    for(auto& g : mediumGroups.GetMap())
    { g.second->Finalize(queueIt.Queue()); queueIt.Next(); }
    for(auto& g : matGroups.GetMap())
    { g.second->Finalize(queueIt.Queue()); queueIt.Next(); }
    for(auto& g : transGroups.GetMap())
    { g.second->Finalize(queueIt.Queue()); queueIt.Next(); }
    for(auto& g : lightGroups.GetMap())
    { g.second->Finalize(queueIt.Queue()); queueIt.Next(); }

    // Pack the surfaces via transform / primitive
    //
    // PG TG {S0,...,SN},  -> AccelGroup (One AccelInstance per S)
    // PG TG {S0,...,SN},  -> AccelGroup
    // ....
    // PG TG {S0,...,SN}   -> AccelGroup
    auto& surfList = surfaces.Vec();
    std::sort(surfList.begin(), surfList.end(), SurfaceLessThan);

    // Do the same thing for lights
    using LightSurfP = Pair<LightSurfaceId, LightSurfaceParams>;
    auto& lSurfList = lightSurfaces.Vec();
    std::sort(lSurfList.begin(), lSurfList.end(), LightSurfaceLessThan);

    // And finally for the cameras as well
    auto& camSurfList = cameraSurfaces.Vec();
    std::sort(camSurfList.begin(), camSurfList.end(), CamSurfaceLessThan);

    // Send it!
    AcceleratorType type = params.accelMode;
    auto accelGenerator = typeGenerators.baseAcceleratorGenerator.at(type);
    auto accelGGeneratorMap = typeGenerators.accelGeneratorMap.at(type);
    auto accelWGeneratorMap = typeGenerators.accelWorkGeneratorMap.at(type);
    if(!accelGenerator || !accelGGeneratorMap || !accelWGeneratorMap)
    {
        throw MRayError("[Tracer]: Unable to find accelerator generators for type \"{}\"",
                        AcceleratorType::ToString(type.type));
    }
    accelerator = accelGenerator.value().get()
    (
        *globalThreadPool, gpuSystem,
        accelGGeneratorMap.value().get(),
        accelWGeneratorMap.value().get()
    );

    // Now partition wrt. Material/Primitive/Transform triplets
    // Almost all of the renderers will require such partitioning.
    // We can give up on the intra surface grouping which was required
    // for accelerators. Material evaluation will only require linear triplets.
    //
    // Flatten the "SurfaceParams"
    // Renderer's will use this to generate "work"
    flattenedSurfaces.reserve(surfaces.Vec().size() *
                              TracerLimits::MaxPrimBatchPerSurface);
    for(const auto& s : surfaces.Vec())
    {
        FlatSurfParams surf = {};
        surf.surfId = s.first;
        surf.tId = s.second.transformId;

        for(uint32_t i = 0; i < s.second.primBatches.size(); i++)
        {
            auto sOut = surf;
            sOut.pId = s.second.primBatches[i];
            sOut.mId = s.second.materials[i];
            flattenedSurfaces.push_back(sOut);
        }
    }
    std::sort(flattenedSurfaces.begin(), flattenedSurfaces.end());

    // Currently none of the groups have a finalize that affect the
    // construction of accelerator(s). However; in future it may be.
    // So again wait the "Finalize" calls of the groups and the texture.
    gpuSystem.SyncAll();

    Timer t;
    t.Start();
    MRAY_LOG("[Tracer]:     Constructing Accelerators ...");
    accelerator->Construct(BaseAccelConstructParams
    {
        .texViewMap = texMem.TextureViews(),
        .primGroups = primGroups.GetMap(),
        .lightGroups = lightGroups.GetMap(),
        .transformGroups = transGroups.GetMap(),
        .mSurfList = surfaces.Vec(),
        .lSurfList = Span<const LightSurfP>(lSurfList)
    });
    t.Lap();
    MRAY_LOG("[Tracer]:     Accelerator construction took {:.2f}ms.",
             t.Elapsed<Millisecond>());

    // Set scene diameter for lights (should only be useful for
    // boundary lights but we try to set for every light)
    AABB3 sceneAABB = accelerator->SceneAABB();
    Vector3 sceneSpan = sceneAABB.GeomSpan();
    // TODO: Should we do this? maximum diameter of some sort?
    //
    // Float sceneDiameter = sceneSpan.Length();
    //
    // Since we are Y up always, get the diameter over XZ plane?
    // Obviously uncommented one is being used
    Float sceneDiameter = Math::Length(Vector2(sceneSpan[0], sceneSpan[2]));
    //
    for(auto& g : lightGroups.GetMap())
        g.second->SetSceneDiameter(sceneDiameter);

    // Rendering will start just after, and it is renderer's responsibility to wait
    // the commit process.
    return SurfaceCommitResult
    {
        .aabb = sceneAABB,
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
                        static_cast<CommonKey>(camSurfId));

    CameraId id = loc->second.cameraId;
    auto key = std::bit_cast<CameraKey>(id);
    auto gId = std::bit_cast<CameraGroupId>(key.FetchBatchPortion());
    auto camGroup = camGroups.at(gId);
    if(!camGroup)
    {
        throw MRayError("Unable to find CameraGroup({})",
                        static_cast<CommonKey>(gId));
    }
    return camGroup.value().get()->AcquireCameraTransform(key);
}

void TracerBase::SetupRenderEnv(TimelineSemaphore* semaphore,
                                uint32_t importAlignment,
                                uint64_t initialAcquireValue)
{
    renderImage = std::make_unique<RenderImage>(semaphore, importAlignment,
                                                initialAcquireValue,
                                                gpuSystem);
}

RendererId TracerBase::CreateRenderer(std::string typeName)
{
    auto rendererGen = typeGenerators.rendererGenerator.at(typeName);
    if(!rendererGen)
    {
        throw MRayError("Unable to find generator for {}",
                        typeName);
    }
    auto rendererWorkPack = typeGenerators.renderWorkGenerator.at(typeName);
    if(!rendererWorkPack)
    {
        throw MRayError("Unable to find work pack generator for {}",
                        typeName);
    }

    auto renderer = rendererGen.value()
    (
        renderImage,
        GenerateTracerView(),
        *globalThreadPool,
        gpuSystem,
        rendererWorkPack.value()
    );
    uint32_t rId = rendererCounter.fetch_add(1u);
    renderers.try_emplace(RendererId(rId), std::move(renderer));
    return RendererId(rId);
}

void TracerBase::DestroyRenderer(RendererId rId)
{
    if(currentRendererId == rId)
    {
        currentRendererId = TracerIdInvalid<RendererId>;
        currentRenderer = nullptr;
    }

    // Remove the current renderer
    if(!renderers.remove_at(rId))
    {
        throw MRayError("Unable to find renderer ({})",
                        static_cast<CommonKey>(rId));
    }

}

void TracerBase::PushRendererAttribute(RendererId rId,
                                       uint32_t attribIndex,
                                       TransientData data)
{
    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<CommonKey>(rId));
    }
    // TODO: Change this
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    renderer.value().get()->PushAttribute(attribIndex, std::move(data),
                                          queue);
}

RenderBufferInfo TracerBase::StartRender(RendererId rId, CamSurfaceId cId,
                                         RenderImageParams rIParams,
                                         Optional<uint32_t> renderLogic0,
                                         Optional<uint32_t> renderLogic1)
{
    // Check render image is setup properly
    if(renderImage.get() == nullptr)
        throw MRayError("Render environment is not set properly! "
                        "Please provide a semaphore to the tracer.");

    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<CommonKey>(rId));
    }
    currentRenderer = renderer.value().get().get();
    currentRendererId = rId;

    return currentRenderer->StartRender(rIParams,
                                        cId,
                                        renderLogic0.value_or(0),
                                        renderLogic1.value_or(0));
}

void TracerBase::SetCameraTransform(RendererId rId, CameraTransform transform)
{
    auto renderer = renderers.at(rId);
    if(!renderer)
    {
        throw MRayError("Unable to find Renderer({})",
                        static_cast<CommonKey>(rId));
    }
    RendererI* rendererPtr = renderer.value().get().get();
    rendererPtr->SetCameraTransform(transform);
}

void TracerBase::StopRender()
{
    if(currentRenderer)
    {
        gpuSystem.SyncAll();
        currentRenderer->StopRender();
    }

}

RendererOutput TracerBase::DoRenderWork()
{
    if(currentRenderer)
        return currentRenderer->DoRender();
    return {};
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

    currentRenderer = nullptr;
    currentRendererId = TracerIdInvalid<RendererId>;

    // GroupId 0 is reserved for some default types,
    // regenerate those
    GenerateDefaultGroups();
}

void TracerBase::Flush() const
{
    gpuSystem.SyncAll();
}

GPUThreadInitFunction TracerBase::GetThreadInitFunction() const
{
    return gpuSystem.GetThreadInitFunction();
}

void TracerBase::SetThreadPool(ThreadPool& tp)
{
    globalThreadPool = &tp;
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
    totalMem += FetchMemUsage(primGroups.GetMap());
    totalMem += FetchMemUsage(camGroups.GetMap());
    totalMem += FetchMemUsage(mediumGroups.GetMap());
    totalMem += FetchMemUsage(matGroups.GetMap());
    totalMem += FetchMemUsage(transGroups.GetMap());
    totalMem += FetchMemUsage(lightGroups.GetMap());
    totalMem += FetchMemUsage(renderers.GetMap());
    totalMem += texMem.GPUMemoryUsage();
    if(accelerator)
        totalMem += accelerator->GPUMemoryUsage();
    return totalMem;
}

const TracerParameters& TracerBase::Parameters() const
{
    return params;
}