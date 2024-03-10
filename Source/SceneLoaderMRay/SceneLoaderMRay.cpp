#include "SceneLoaderMRay.h"

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "Core/Timer.h"
#include "Core/NormTypes.h"

#include "MeshLoader/EntryPoint.h"
#include "MeshLoaderJson.h"

#include "ImageLoader/EntryPoint.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>
#include <barrier>
#include <atomic>
#include <istream>

#include "JsonNode.h"



template<class GroupId>
Optional<TransientData> GenericAttributeLoad(const JsonNode& node,
                                             const GenericAttributeInfo& attribInfo)
{
    Optional<TransientData> result;

    using enum GenericAttributeInfo::E;
    std::string name = std::get<LOGIC_INDEX>(attrib);
    MRayDataTypeRT layout = std::get<LAYOUT_INDEX>(attrib);
    bool isArray = (std::get<IS_ARRAY_INDEX>(attrib) ==
                    AttributeIsArray::IS_ARRAY);
    bool isOptional = (std::get<OPTIONALITY_INDEX>(attrib) ==
                        AttributeOptionality::MR_OPTIONAL);

    auto r = std::visit([&](auto&& dataType) -> Optional<TransientData>
    {
        using T = std::remove_cvref_t<decltype(dataType)>::Type;

        if(isArray && isOptional)
        {
            return node.AccessOptionalDataArray<T>(name);
        }
        else if(isArray && !isOptional)
        {
            return node.AccessDataArray<T>(name);
        }
        else if(!isArray && isOptional)
        {
            TransientData result(std::in_place_type_t<T>{}, 1);
            Optional<T> r = node.AccessOptionalData<T>(name);
            if(!r.has_value()) return std::nullopt;

            T val = r.value();
            result.Push<T>(Span<const T, 1>(&val, 1));
            return std::move(result);
        }
        else
        {
            TransientData result(std::in_place_type_t<T>{}, 1);
            T val = node.AccessData<T>(name);
            result.Push<T>(Span<const T, 1>(&val, 1));
            return std::move(result);
        }
    }, layout);

    return r;
}

template<class AttributeInfoType>
void GenericJsonLoad(TransientData& dataOut,
                     std::vector<Optional<NodeTexStruct>>& texOut,
                     const AttributeInfoType& attrib, const JsonNode& node)
{
    using enum AttributeInfoType::E;
    std::string name = std::get<LOGIC_INDEX>(attrib);
    MRayDataTypeRT layout = std::get<LAYOUT_INDEX>(attrib);
    bool isArray = (std::get<IS_ARRAY_INDEX>(attrib) ==
                    AttributeIsArray::IS_ARRAY);
    bool isOptional = (std::get<OPTIONALITY_INDEX>(attrib) ==
                        AttributeOptionality::MR_OPTIONAL);
    bool isTexturable = false;
    if constexpr(std::is_same_v<AttributeInfoType, TexturedAttributeInfo>)
        isTexturable = (std::get<TEXTURABLE_INDEX>(attrib) !=
                        AttributeTexturable::MR_CONSTANT_ONLY);

    bool isOptionalTexture = false;
    if constexpr(std::is_same_v<AttributeInfoType, TexturedAttributeInfo>)
        isOptionalTexture = (std::get<TEXTURABLE_INDEX>(attrib) ==
                             AttributeTexturable::MR_OPTIONAL_TEXTURE);

    auto r = std::visit([&](auto&& dataType) -> void
    {
        using T = std::remove_cvref_t<decltype(dataType)>::Type;

        if(isTexturable)
        {
            using NTS = NodeTexStruct;
            if(isOptionalTexture)
                texOut.push_back(node.AccessOptionalTexture(name));
            else
            {
                Variant<NTS, T> result = node.AccessTexturableData<T>(name);
                if(std::holds_alternative<T>())
                {
                    texOut.push_back(std::nullopt);
                    dataOut.Push(Span<const T>(&std::get<T>(result), 1))
                }
                else
                {
                    T _{};
                    texOut.push_back(std::get<NTS>(result));
                    dataOut.Push(Span<const T>(&_, 1))
                }
            }
        }
        else if(isArray)
        {
            Optional<TransientData> result;
            result = (isOptional)
                        ? node.AccessOptionalDataArray<T>(name)
                        : node.AccessDataArray<T>(name);
            if(result.has_value())
                dataOut.Push(ToConstSpan(ToSpan<T>(result.value())));
        }
        else
        {
            Optional<T> result;
            result = (isOptional)
                        ? node.AccessOptionalData<T>(name)
                        : node.AccessData<T>(name);
            if(result.has_value())
                dataOut.Push<T>(Span<const T>(&result.value(), 1));
        }
    }, layout);
}

void LoadPrimitive(TracerI& tracer,
                   PrimGroupId groupId,
                   PrimBatchId batchId,
                   uint32_t meshInternalIndex,
                   const MeshFileI* meshFile)
{
    using enum PrimitiveAttributeLogic;
    const auto& attributeList = tracer.AttributeInfo(groupId);

    for(uint32_t attributeIndex = 0;
        attributeIndex < attributeList.size();
        attributeIndex++)
    {
        const auto& attribute = attributeList[attributeIndex];
        PrimitiveAttributeLogic attribLogic = std::get<PrimAttributeInfo::LOGIC_INDEX>(attribute);
        AttributeOptionality optionality = std::get<PrimAttributeInfo::OPTIONALITY_INDEX>(attribute);
        MRayDataTypeRT groupsLayout = std::get<PrimAttributeInfo::LAYOUT_INDEX>(attribute);
        MRayDataTypeRT filesLayout = meshFile->AttributeLayout(attribLogic,
                                                               meshInternalIndex);

        // Is this data available?
        if(!meshFile->HasAttribute(attribLogic))
        {
            if(optionality == AttributeOptionality::MR_MANDATORY)
                throw MRayError("Mesh File{:s}:[{:d}] do not have \"{}\""
                                "which is mandatory for {}",
                                meshFile->Name(), meshInternalIndex,
                                PrimAttributeStringifier::ToString(attribLogic),
                                tracer.TypeName(groupId));
            continue;
        }
        // Special case for default triangle
        if(attribLogic == NORMAL && groupsLayout.Name() == MRayDataEnum::MR_QUATERNION &&
           meshFile->HasAttribute(TANGENT) && meshFile->HasAttribute(BITANGENT) &&
           meshFile->AttributeLayout(TANGENT).Name() == MRayDataEnum::MR_VECTOR_3 &&
           meshFile->AttributeLayout(BITANGENT).Name() == MRayDataEnum::MR_VECTOR_3 &&
           meshFile->AttributeLayout(NORMAL).Name() == MRayDataEnum::MR_VECTOR_3)
        {
            size_t normalCount = meshFile->MeshAttributeCount(meshInternalIndex);
            TransientData q(std::in_place_type_t<Quaternion>{}, normalCount);
            Span<Quaternion> quats = q.AccessAs<Quaternion>();

            // Utilize TBN matrix directly
            TransientData t = meshFile->GetAttribute(TANGENT, meshInternalIndex);
            TransientData b = meshFile->GetAttribute(BITANGENT, meshInternalIndex);
            TransientData n = meshFile->GetAttribute(attribLogic, meshInternalIndex);

            Span<const Vector3> tangents    = t.AccessAs<const Vector3>();
            Span<const Vector3> bitangents  = b.AccessAs<const Vector3>();
            Span<const Vector3> normals     = n.AccessAs<const Vector3>();

            for(size_t i = 0; i < quats.size(); i++)
            {
                quats[i] = TransformGen::ToSpaceQuat(tangents[i],
                                                     bitangents[i],
                                                     normals[i]);
            }
            tracer.PushPrimAttribute(batchId, attributeIndex, std::move(q));
        }
        // Is this data's layout match with the primitive group
        else if(groupsLayout.Name() != filesLayout.Name())
        {
            // TODO: Add CPU conversion logic here for basic conversions
            // (Both is floating point type and conversion is *not* narrowing)
            // For the rest below error should be fine

            // We require exact match currently
            throw MRayError("Mesh File{:s}:[{:d}]'s data layout of \"{}\""
                            "(has type{:s}) does not match the {}'s data layout "
                            "(which is {:s})",
                            meshFile->Name(), meshInternalIndex,
                            PrimAttributeStringifier::ToString(attribLogic),
                            MRayDataTypeStringifier::ToString(filesLayout.Name()),
                            tracer.TypeName(groupId),
                            MRayDataTypeStringifier::ToString(groupsLayout.Name()));
        }
        // All Good, load and send
        else
        {
            tracer.PushPrimAttribute(batchId, attributeIndex,
                                     meshFile->GetAttribute(attribLogic, meshInternalIndex));
        }
    }
}

void SceneLoaderMRay::ExceptionList::AddException(MRayError&& err)
{
    size_t location = size.fetch_add(1);
    // If too many exceptions skip it
    if(location < MaxExceptionSize)
        exceptions[location] = std::forward<MRayError>(err);
}

std::string SceneLoaderMRay::SceneRelativePathToAbsolute(std::string_view sceneRelativePath,
                                                         std::string_view scenePath)
{
    using namespace std::filesystem;
    path fullPath = path(scenePath) / path(sceneRelativePath);
    return absolute(fullPath).string();
}

LightSurfaceStruct SceneLoaderMRay::LoadBoundary(const nlohmann::json& n)
{
    LightSurfaceStruct boundary = n.get<LightSurfaceStruct>();
    if(boundary.lightId == std::numeric_limits<uint32_t>::max())
        throw MRayError("Boundary light must be set!");
    if(boundary.mediumId == std::numeric_limits<uint32_t>::max())
        throw MRayError("Boundary medium must be set!");
    if(boundary.transformId)
        throw MRayError("Boundary transform must be set!");
    return boundary;
}

std::vector<SurfaceStruct> SceneLoaderMRay::LoadSurfaces(const nlohmann::json& nArray)
{
    std::vector<SurfaceStruct> result;
    for(const auto& n : nArray)
    {
        result.push_back(n.get<SurfaceStruct>());
    }
    return result;
}

std::vector<CameraSurfaceStruct> SceneLoaderMRay::LoadCamSurfaces(const nlohmann::json& nArray,
                                                                  uint32_t boundaryMediumId)
{

    std::vector<CameraSurfaceStruct> result;
    for(const auto& n : nArray)
    {
        result.push_back(n.get<CameraSurfaceStruct>());
        if(result.back().mediumId == std::numeric_limits<uint32_t>::max())
            result.back().mediumId = boundaryMediumId;
    }
    return result;
}

std::vector<LightSurfaceStruct> SceneLoaderMRay::LoadLightSurfaces(const nlohmann::json& nArray,
                                                                   uint32_t boundaryMediumId)
{
    std::vector<LightSurfaceStruct> result;
    for(const auto& n : nArray)
    {
        result.push_back(n.get<LightSurfaceStruct>());
        if(result.back().mediumId == std::numeric_limits<uint32_t>::max())
            result.back().mediumId = boundaryMediumId;
    }
    return result;
}

void SceneLoaderMRay::DryRunLightsForPrim(std::vector<uint32_t>& primIds,
                                          const TypeMappedNodes& lightNodes,
                                          const TracerI& tracer)
{
    for(const auto& l : lightNodes)
    {
        LightAttributeInfoList lightAttributes = tracer.LightAttributeInfo(l.first);
        if(l.first != NodeNames::LIGHT_TYPE_PRIMITIVE)
            continue;
        // Light Type is primitive, it has to have "primitive" field
        for(const auto& node : l.second)
        {
            uint32_t primId = node.AccessData<uint32_t>(NodeNames::PRIMITIVE);
            primIds.push_back(primId);
        }
    }
}

template <class TracerInterfaceFunc>
void SceneLoaderMRay::DryRunNodesForTex(std::vector<NodeTexStruct>& textureIds,
                                        const TypeMappedNodes& nodes,
                                        const TracerI& tracer,
                                        TracerInterfaceFunc&& func)
{
    for(const auto& n : nodes)
    {
        TexturedAttributeInfoList texAttributes = std::invoke(func, tracer, n.first);
        // Light Type is primitive, it has to have "primitive" field
        for(const auto& node : n.second)
        for(const auto& att : texAttributes)
        {
            AttributeTexturable texturable = std::get<MatAttributeInfo::TEXTURABLE_INDEX>(att);
            AttributeOptionality optional = std::get<MatAttributeInfo::OPTIONALITY_INDEX>(att);
            std::string_view name = std::get<MatAttributeInfo::LOGIC_INDEX>(att);
            if(texturable == AttributeTexturable::MR_CONSTANT_ONLY)
                continue;

            if(texturable == AttributeTexturable::MR_TEXTURE_ONLY)
            {
                if(optional == AttributeOptionality::MR_OPTIONAL)
                {
                    auto ts = node.AccessOptionalData<NodeTexStruct>(name);
                    if(ts.has_value()) textureIds.push_back(ts.value());
                }
                else
                {
                    auto ts = node.AccessData<NodeTexStruct>(name);
                    textureIds.push_back(ts);
                }
            }
            else if(texturable == AttributeTexturable::MR_TEXTURE_OR_CONSTANT)
            {
                MRayDataTypeRT dataType = std::get<MatAttributeInfo::LAYOUT_INDEX>(att);
                std::visit([&node, name, &textureIds](auto&& dataType)
                {
                    using T = std::remove_cvref_t<decltype(dataType)>::Type;
                    auto value = node.AccessTexturableData<T>(name);
                    if(std::holds_alternative<NodeTexStruct>(value))
                        textureIds.push_back(std::get<NodeTexStruct>(value));
                }, dataType);
            }
        }
    }
}

template<class Loader, class IdType>
void LoadGroups(typename SceneLoaderMRay::MutexedMap<std::map<uint32_t, IdType>>& outputMappings,
                typename SceneLoaderMRay::ExceptionList& exceptions,
                const typename SceneLoaderMRay::TypeMappedNodes& nodes,
                BS::thread_pool& threadPool,
                Loader&& loader)
{
    using GroupIdType   = typename Loader::GroupIdType;
    using IdPair        = Pair<uint32_t, IdType>;
    using PerGroupList  = std::vector<IdPair>;
    using IdList        = std::vector<IdType>;

    for(const auto& [typeName, nodes] : nodeLists)
    {
        const uint32_t groupEntityCount = static_cast<uint32_t>(nodes.size());
        auto groupEntityList = std::make_shared<PerGroupList>(groupEntityCount);

        GroupIdType groupId = loader.CreateGroup(std::string(typeName));

        // Construct Barrier
        auto BarrierFunc = [groupId, &loader]() noexcept
        {
            // When barrier completed
            // Reserve the space for mappings
            // Commit group reservations
            loader.CommitReservations(groupId);
        };
        // Determine the thread size
        uint32_t threadCount = std::min(threadPool.get_thread_count(),
                                        groupEntityCount);

        using Barrier = std::barrier<decltype(BarrierFunc)>;
        auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

        auto future = threadPool.submit_blocks(std::size_t(0), std::size_t(groupEntityCount),
        // Copy the shared pointers, capture by reference the rest
        [&, barrier, groupEntityList](size_t start, size_t end)
        {
            size_t localCount = start - end;
            auto nodeRange = Span<const JsonNode>(nodes.cbegin() + start, localCount);
            IdList generatedIds;
            try
            {
                generatedIds = loader.ReserveEntities(groupId, nodeRange);
                for(size_t i = start; i < end; i++)
                {
                    size_t localI = i - start;
                    auto& groupList = *groupEntityList;
                    groupList[i] = std::make_pair(nodes[i].Id(),
                                                  generatedIds[localI]);
                }
            }
            catch(MRayError& e)
            {
                exceptions.AddException(std::move(e));
                barrier->arrive_and_drop();
                return;
            }

            // Commit barrier
            barrier->arrive_and_wait();
            // Group is committed, now we can issue writes
            try
            {
                const auto& groupList = *groupEntityList;
                loader.LoadEntities(groupId, nodeRange, generatedIds);
            }
            catch(MRayError& e)
            {
                exceptions.AddException(std::move(e));
            }
        }, threadCount);

        // Move future to shared_ptr
        using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
        FutureSharedPtr futureShared = std::make_shared<BS::multi_future<void>>(std::move(future));

        // Issue a one final task that pushes the primitives to the global map
        threadPool.detach_task([&, future = futureShared, groupEntityList]()
        {
            // Wait other tasks to complere
            future->wait();
            // After this point groupBatchList is fully loaded
            std::scoped_lock lock(outputMappings.mutex);
            outputMappings.map.insert(groupEntityList->cbegin(), groupEntityList->cend());
        });
    }
}

void SceneLoaderMRay::LoadTextures(TracerI& tracer, ExceptionList& exceptions)
{
    using TextureIdList = std::vector<std::pair<NodeTexStruct, TextureId>>;

    // Construct Image Loader
    std::shared_ptr<ImageLoaderI> imgLoader = CreateImageLoader();
    auto texIdListPtr = std::make_shared<TextureIdList>(textureNodes.size());

    // Flatten the std::map (we can't multi-thread it efficiently)
    // TODO: Ensure uniqueness without a std::map maybe? (Flatmap?)
    std::vector<std::pair<NodeTexStruct, JsonNode>> flattenedTexMap;
    flattenedTexMap.reserve(textureNodes.size());
    for(const auto& [texStruct, jsonNode] : textureNodes)
        flattenedTexMap.emplace_back(texStruct, jsonNode);

    // Issue loads to the thread pool
    auto BarrierFunc = [&tracer]() noexcept
    {
        // When barrier completed
        // Reserve the space for mappings
        // Commit textures greoups reservations
        tracer.CommitTextures();
    };

    // Determine the thread size
    uint32_t threadCount = std::min(threadPool.get_thread_count(),
                                    static_cast<uint32_t>(flattenedTexMap.size()));

    using Barrier = std::barrier<decltype(BarrierFunc)>;
    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

    auto future = threadPool.submit_blocks(std::size_t(0), flattenedTexMap.size(),
    // Copy the shared pointers, capture by reference the rest
    [&, texIdListPtr, barrier](size_t start, size_t end)
    {
        // TODO: check if we twice opening is bottleneck?
        // We are opening here to determining size/format
        // and on the other iteration we actual memcpy it
        try
        {
            for(size_t i = start; i < end; i++)
            {
                const auto& [texStruct, jsonNode] = flattenedTexMap[i];
                auto fileName = jsonNode.AccessData<std::string>(NodeNames::TEX_NODE_FILE);
                fileName = SceneLoaderMRay::SceneRelativePathToAbsolute(fileName, scenePath);

                // Currently no flags are utilized on header load time
                // TODO: Check here if this fails
                Expected<ImageHeader<2>> headerE = imgLoader->ReadImageHeader2D(fileName);
                if(!headerE.has_value())
                {
                    exceptions.AddException(std::move(headerE.error()));
                    barrier->arrive_and_drop();
                    return;
                }

                // Always expand to 4 channel due to HW limitation
                headerE.value().pixelType = ImageLoaderI::TryExpandTo4CFormat(headerE.value().pixelType);

                const auto& header = headerE.value();
                TextureId tId = tracer.CreateTexture2D(header.dimensions,
                                                       header.mipCount,
                                                       header.pixelType.Name());

                auto& texIdList = *texIdListPtr;
                texIdList[i] = std::make_pair(texStruct, tId);
            }
        }
        catch(MRayError& e)
        {
            exceptions.AddException(std::move(e));
            barrier->arrive_and_drop();
            return;
        }
        // Barrier code is invoked, and all textures are allocated
        barrier->arrive_and_wait();

        try
        {
            for(size_t i = start; i < end; i++)
            {
                const auto& [texStruct, jsonNode] = flattenedTexMap[i];
                bool loadAsSigned = jsonNode.AccessData<bool>(NodeNames::TEX_NODE_AS_SIGNED);
                bool isData = jsonNode.AccessData<bool>(NodeNames::TEX_NODE_IS_DATA);
                auto fileName = jsonNode.AccessData<std::string>(NodeNames::TEX_NODE_FILE);
                fileName = SceneLoaderMRay::SceneRelativePathToAbsolute(fileName, scenePath);

                using enum ImageIOFlags::F;
                ImageIOFlags flags;
                flags[DISREGARD_COLOR_SPACE] = isData;
                flags[LOAD_AS_SIGNED] = loadAsSigned;
                flags[TRY_3C_4C_CONVERSION] = true;     // Always do channel expand (HW limitation)

                Expected<Image<2>> imgE = imgLoader->ReadImage2D(fileName, flags);
                if(!imgE.has_value())
                {
                    exceptions.AddException(std::move(imgE.error()));
                    return;
                }

                auto& img = imgE.value();
                // Send data mip by mip
                for(uint32_t j = 0; j < img.header.mipCount; i++)
                {
                    auto& texIdList = *texIdListPtr;
                    tracer.PushTextureData(texIdList[i].second, j,
                                           std::move(img.imgData[j].pixels));
                }
            }
        }
        catch(MRayError& e)
        {
            exceptions.AddException(std::move(e));
            return;
        }
    }, threadCount);

    // Move the future to shared ptr
    using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
    FutureSharedPtr futureShared = std::make_shared<BS::multi_future<void>>(std::move(future));

    // Issue a one final task that pushes the primitives to the global map
    threadPool.detach_task([&, this, future = futureShared, texIdListPtr]()
    {
        // Wait other tasks to complere
        future->wait();
        // Thread Generated Textures are finalized
        for(const auto& pair : (*texIdListPtr))
            texMappings.emplace(pair.first, pair.second);
    });
    // All important data is in shared_ptrs we can safely exit scope.
}

void SceneLoaderMRay::LoadMediums(TracerI& tracer, ExceptionList& exceptions)
{
    struct AttribData
    {
        bool isTextured;
        bool isOptional;
        bool isArray;
        TransientData d;
        std::vector<Optional<NodeTexStruct>> a;
    };

    // Rare usage of in-function class
    struct MediumLoader
    {
        public:
        using GroupIdType = MediumGroupId;

        private:
        TracerI&            tracer;


        public:
        MediumLoader(TracerI& t) : tracer(t) {}
        MediumGroupId CreateGroup(const std::string gn)
        {
            return tracer.CreateMediumGroup(gn);
        }
        void CommitReservations(MediumGroupId groupId)
        {
            return tracer.CommitMediumReservations(groupId);
        }
        MediumIdList ReserveEntities(MediumGroupId groupId, Span<const JsonNode> nodes)
        {
            return tracer.ReserveMediums(groupId, nodes.size());
        }
        void LoadEntitiy(MediumGroupId groupId,
                         Span<const JsonNode> nodes,
                         const MediumIdList& idList)
        {
            assert(nodes.size() == idList.size());
            MediumAttributeInfoList list = tracer.AttributeInfo(groupId);
            std::vector<AttribData> data(list.size());
            for(size_t i = 0; i < nodes.size(); i++)
            {
                const JsonNode& node = nodes[i];
                MediumId id = idList[i];

                for(const MediumAttributeInfo& attrib : list)
                {
                    GenericJsonLoad()...;
                }

            }

            tracer.Push
        }
    };

    LoadGroups(mediumMappings, exceptions, mediumNodes,
               threadPool, MediumLoader(tracer));
}

void SceneLoaderMRay::LoadMaterials(TracerI& tracer, ExceptionList& exceptions)
{
    //// Issue loads to the thread pool
    //for(const auto& m : materialNodes)
    //{
    //    uint32_t groupMatCount = static_cast<uint32_t>(m.second.size());
    //    using GroupMaterialList = std::vector<Pair<uint32_t, MaterialId>>;
    //    auto groupMatList = std::make_shared<GroupMaterialList>(groupMatCount);

    //    std::string_view matTypeName = m.first;
    //    MatGroupId groupId = tracer.CreateMaterialGroup(std::string(matTypeName));

    //    // Construct Barrier
    //    auto BarrierFunc = [groupId, groupMatList, &tracer]() noexcept
    //    {
    //        // When barrier completed
    //        // Reserve the space for mappings
    //        // Commit mat group reservations
    //        tracer.CommitMatReservations(groupId);
    //    };
    //    // Determine the thread size
    //    uint32_t threadCount = std::min(threadPool.get_thread_count(),
    //                                    groupMatCount);

    //    using Barrier = std::barrier<decltype(BarrierFunc)>;
    //    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

    //    auto future = threadPool.submit_blocks(std::size_t(0), m.second.size(),
    //    // Copy the shared pointers, capture by reference the rest
    //    [&, this, barrier, groupMatList](size_t start, size_t end)
    //    {
    //        try
    //        {
    //            for(size_t i = start; i < end; i++)
    //            {
    //                //// Each material optionally express medium interface
    //                //// that are seperating???
    //                //using namespace NodeNames;
    //                //const JsonNode& node = m.second[i];
    //                //auto mediumFront = node.AccessOptionalData<uint32_t>(MAT_NODE_MEDIUM_FRONT);
    //                //auto mediumBack = node.AccessOptionalData<uint32_t>(MAT_NODE_MEDIUM_BACK);

    //                //mediumMappings.map.

    //                //MaterialId mId = tracer.ReserveMaterial(groupId,
    //                //                                        );
    //                //(*groupMatList)[i] = std::make_pair(nodeId(), mId);

    //            }
    //        }
    //        catch(MRayError& e)
    //        {
    //            exceptions.AddException(std::move(e));
    //            barrier->arrive_and_drop();
    //            return;
    //        }

    //        // Commit barrier
    //        barrier->arrive_and_wait();
    //        // Material Group is committed, we can issue writes to it now
    //        try
    //        {
    //            for(size_t i = start; i < end; i++)
    //            {
    //                //..............
    //            }
    //        }
    //        catch(MRayError& e)
    //        {
    //            exceptions.AddException(std::move(e));
    //        }
    //    }, threadCount);

    //    // Move future to shared_ptr
    //    using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
    //    FutureSharedPtr futureShared = std::make_shared<BS::multi_future<void>>(std::move(future));

    //    // Issue a one final task that pushes the primitives to the global map
    //    threadPool.detach_task([&, this, future = futureShared, groupMatList]()
    //    {
    //        // Wait other tasks to complere
    //        future->wait();
    //        // After this point groupBatchList is fully loaded
    //        std::scoped_lock lock(matMappings.mutex);
    //        matMappings.map.insert(groupMatList->flatMaterials.cbegin(),
    //                               groupMatList->flatMaterials.cend());
    //    });
    //}
}

void SceneLoaderMRay::LoadTransforms(TracerI& tracer, ExceptionList& exceptions)
{
    struct A
    {

    };

    thread_local uint32_t localAttributeSize = 0;


    // Rare usage of in-function class
    struct TransformLoader
    {
        public:
        using GroupIdType = TransGroupId;

        private:
        TracerI& tracer;

        public:
        TransformLoader(TracerI& t) : tracer(t) {}
        TransGroupId CreateGroup(const std::string gn)
        {
            return tracer.CreateTransformGroup(gn);
        }
        void CommitReservations(TransGroupId groupId)
        {
            return tracer.CommitTransReservations(groupId);
        }
        TransformIdList ReserveEntities(TransGroupId groupId,
                                        Span<const JsonNode> nodes)
        {
            std::vector<uint32_t> counts;
            counts.reserve(nodes.size());
            // Find the first arrayed type in nodes
            // Accumulate the type
            // Reserve transforms accordingly
            TransAttributeInfoList list = tracer.AttributeInfo(groupId);
            for(const auto& l : list)
            {
                using enum AttributeIsArray;
                using enum TransAttributeInfo::E;
                if(std::get<IS_ARRAY_INDEX>(l) != IS_ARRAY)
                    continue;
                std::string_view name = std::get<LOGIC_INDEX>(l);

                for(const JsonNode& node : nodes)
                {
                    size_t count = static_cast<uint32_t>(node.AccessDataArraySize(name));
                    counts.push_back(count);
                    localAttributeSize += count;
                }
                break;
            }
            return tracer.ReserveTransformations(groupId, counts);
        }
        void LoadEntitiy(TransGroupId groupId, TransformId id, const JsonNode& node)
        {
            std::vector<Optional<TransientData>> attributeData = GenericLoad(tracer,
                                                                             groupId,
                                                                             node);
            uint32_t attributeIndex = 0;
            for(auto& data : attributeData)
            {
                if(!data.has_value()) continue;

                Vector2ui range(static_cast<uint32_t>(id),
                                static_cast<uint32_t>(id) + 1);

                tracer.PushTransAttribute(groupId, range,
                                          attributeIndex,
                                          std::move(data.value()));
                attributeIndex++;
            }

        }
    };

    LoadGroups(transformMappings, exceptions,
               transformNodes, threadPool, TransformLoader(tracer));
}

void SceneLoaderMRay::LoadPrimitives(TracerI& tracer, ExceptionList& exceptions)
{
    using PrimGroupBatchList = std::vector<Pair<uint32_t, PrimBatchId>>;
    std::shared_ptr<const MeshLoaderPoolI> meshLoaderPool = CreateMeshLoaderPool();

    // Issue loads to the thread pool
    for(const auto& m : primNodes)
    {
        uint32_t groupBatchCount = static_cast<uint32_t>(m.second.size());
        auto groupBatchList = std::make_shared<PrimGroupBatchList>(groupBatchCount);

        std::string_view primTypeName = m.first;
        PrimGroupId groupId = tracer.CreatePrimitiveGroup(std::string(primTypeName));

        // Construct Barrier
        auto BarrierFunc = [groupId, groupBatchList, &tracer]() noexcept
        {
            // When barrier completed
            // Reserve the space for mappings
            // Commit prim groups reservations
            tracer.CommitPrimReservations(groupId);
        };
        // Determine the thread size
        uint32_t threadCount = std::min(threadPool.get_thread_count(),
                                        groupBatchCount);

        using Barrier = std::barrier<decltype(BarrierFunc)>;
        auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

        auto future = threadPool.submit_blocks(std::size_t(0), m.second.size(),
        // Copy the shared pointers, capture by reference the rest
        [&, this, barrier, meshLoaderPool, groupBatchList](size_t start, size_t end)
        {
            size_t localBatchCount = end - start;

            // TODO: Too many micro allocations due to map
            // revise over this.
            //
            // Per "dll" loaders
            // Key is a "tag" entry on the json. For example, "assimp" tag
            // will call assimp mesh loader. If "assimp" does not support .obj for example
            // loader will generate an error.
            //
            // The reason for this approach is to provide a utility for user to add new dlls
            // for specific file extensions via per mesh file basis.
            // If assimp "sucks" for new fbx files, however robustly works on old fbx files
            // user can write new loader with a tag and load these files.
            std::map<std::string, std::unique_ptr<MeshLoaderI>> loaders;
            // Mesh files that opened via some loader
            // This is here unfortunately as a limitation/functionality of the assimp.
            // Assimp can post process meshes crates tangents/optimization etc. This means
            // we cannot pre-determine the size of the vertices/indices just by looking a some
            // form of header on the file. So we load the entire mesh and store before the commit
            // of the primitive group.
            //
            // Key is the full path of the mesh file. For in node primitives,
            //  it is the scene "primitiveId".
            std::map<std::string, std::unique_ptr<MeshFileI>> meshFiles;
            // Each mesh may have multiple submeshes so we don't wastefully open the same file
            // multiple times
            std::vector<Pair<uint32_t, const MeshFileI*>> batchFiles;
            batchFiles.reserve(localBatchCount);

            try
            {
                for(size_t i = start; i < end; i++)
                {
                    const JsonNode& node = m.second[i];
                    std::string tag = std::string(node.Tag());

                    uint32_t innerIndex = 0;
                    const MeshFileI* meshFile = nullptr;

                    if(tag == NodeNames::NODE_PRIM_TRI ||
                       tag == NodeNames::NODE_PRIM_TRI_INDEXED)
                    {
                        using namespace NodeNames;
                        bool isIndexed = (tag == NODE_PRIM_TRI_INDEXED);
                        auto meshFilePtr = std::make_unique<JsonTriangle>(m.second[i], isIndexed);
                        meshFile = meshFiles.emplace(std::to_string(m.second[i].Id()),
                                                     std::move(meshFilePtr)).first->second.get();
                    }
                    else if(tag == NodeNames::NODE_PRIM_SPHERE)
                    {
                        using namespace NodeNames;
                        auto meshFilePtr = std::make_unique<JsonSphere>(m.second[i]);
                        meshFile = meshFiles.emplace(std::to_string(m.second[i].Id()),
                                                     std::move(meshFilePtr)).first->second.get();
                    }
                    else
                    {
                        std::string fileName = node.CommonData<std::string>(NodeNames::FILE);
                        fileName = SceneLoaderMRay::SceneRelativePathToAbsolute(fileName, scenePath);
                        innerIndex = node.AccessData<uint32_t>(NodeNames::INNER_INDEX);

                        // Find a Loader
                        auto r0 = loaders.emplace(tag, nullptr);
                        if(!r0.second) r0.first->second = meshLoaderPool->AcquireALoader(tag);
                        const auto& meshLoader = r0.first->second;

                        // Find mesh file
                        // TODO: this is slow probably due to long file name as key
                        auto r1 = meshFiles.emplace(fileName, nullptr);
                        if(!r1.second) r1.first->second = meshLoader->OpenFile(fileName);
                        meshFile = r1.first->second.get();
                    }

                    // Finally Reserve primitives
                    PrimCount pc
                    {
                        .primCount = meshFile->MeshPrimitiveCount(innerIndex),
                        .attributeCount = meshFile->MeshAttributeCount(innerIndex)
                    };
                    PrimBatchId tracerId = tracer.ReservePrimitiveBatch(groupId, pc);
                    (*groupBatchList)[i] = std::make_pair(node.Id(), tracerId);
                    batchFiles.emplace_back(innerIndex, meshFile);
                }
            }
            catch(MRayError& e)
            {
                exceptions.AddException(std::move(e));
                barrier->arrive_and_drop();
                return;
            }

            // Commit barrier
            barrier->arrive_and_wait();
            // Primitive Group is committed, we can issue writes to it now
            try
            {
                for(size_t i = start; i < end; i++)
                {
                    size_t localIndex = end - i;
                    const auto& [innerIndex, meshFile] = batchFiles[localIndex];
                    PrimBatchId batchId = (*groupBatchList)[i].second;
                    LoadPrimitive(tracer, groupId, batchId,
                                  innerIndex, meshFile);
                }
            }
            catch(MRayError& e)
            {
                exceptions.AddException(std::move(e));
            }
        }, threadCount);

        // Move future to shared_ptr
        using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
        FutureSharedPtr futureShared = std::make_shared<BS::multi_future<void>>(std::move(future));

        // Issue a one final task that pushes the primitives to the global map
        threadPool.detach_task([&, this, future = futureShared, groupBatchList]()
        {
            // Wait other tasks to complere
            future->wait();
            // After this point groupBatchList is fully loaded
            std::scoped_lock lock(primMappings.mutex);
            primMappings.map.insert(groupBatchList->cbegin(),
                                    groupBatchList->cend());
        });
    }
}

void SceneLoaderMRay::LoadCameras(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::LoadLights(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::CreateTypeMapping(const TracerI& tracer,
                                        const SceneSurfList& surfaces,
                                        const SceneCamSurfList& camSurfaces,
                                        const SceneLightSurfList& lightSurfaces,
                                        const LightSurfaceStruct& boundary)
{
    // Given N definition items, and M references on those items
    // where M >= N, create a map of common definitions -> referred definition list.

    // Definition items assumed to be random. Worst case this is O(N * M).
    // On a proper scene with transforms, M >> N. But N >> M can be possible if we
    // allow including multiple definition arrays (and some form of #include equavilence)
    //
    // This implementations assumes the first case is most common. So we create a HT
    // of N's and their locations. Then iterate over the M's and create the type
    constexpr uint32_t ARRAY_INDEX = 0;
    constexpr uint32_t INNER_INDEX = 1;
    using ItemLocation = Pair<uint32_t, uint32_t>;
    using ItemLocationMap = std::unordered_map<uint32_t, ItemLocation>;

    auto CreateHT = [](ItemLocationMap& result,
                       const nlohmann::json& definitions) -> void
    {
        for(uint32_t i = 0; i < definitions.size(); i++)
        {
            const auto& node = definitions[i];
            const auto idNode = node.at(NodeNames::ID);
            ItemLocation itemLoc;
            std::get<ARRAY_INDEX>(itemLoc) = i;
            if(!idNode.is_array())
            {
                std::get<INNER_INDEX>(itemLoc) = 0;
                result.emplace(idNode.get<uint32_t>(), itemLoc);
            }
            else for(uint32_t j = 0; j < idNode.size(); j++)
            {
                const auto& id = idNode[j];
                std::get<INNER_INDEX>(itemLoc) = j;
                result.emplace(id.get<uint32_t>(), itemLoc);
            }
        }
    };

    using namespace TracerConstants;
    // Prims
    ItemLocationMap primHT;
    primHT.reserve(surfaces.size() * MaxPrimBatchPerSurface +
                   lightSurfaces.size());
    std::future<void> primHTReady = threadPool.submit_task(
    [&primHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(primHT, sceneJson.at(NodeNames::PRIMITIVE_LIST));
    });
    // Materials
    ItemLocationMap matHT;
    matHT.reserve(surfaces.size() * MaxPrimBatchPerSurface);
    std::future<void> matHTReady = threadPool.submit_task(
    [&matHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(matHT, sceneJson.at(NodeNames::MATERIAL_LIST));
    });
    // Cameras
    ItemLocationMap camHT;
    camHT.reserve(camSurfaces.size());
    std::future<void> camHTReady = threadPool.submit_task(
    [&camHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(camHT, sceneJson.at(NodeNames::CAMERA_LIST));
    });
    // Lights
    // +1 Comes from boundary light
    ItemLocationMap lightHT;
    lightHT.reserve(lightSurfaces.size() + 1);
    std::future<void> lightHTReady = threadPool.submit_task(
    [&lightHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(lightHT, sceneJson.at(NodeNames::LIGHT_LIST));
    });
    // Transforms
    ItemLocationMap transformHT;
    transformHT.reserve(lightSurfaces.size() +
                        surfaces.size() +
                        camSurfaces.size() + 1);
    std::future<void> transformHTReady = threadPool.submit_task(
    [&transformHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(transformHT, sceneJson.at(NodeNames::TRANSFORM_LIST));
    });

    // Mediums
    // Medium worst case is practically impossible
    // Each surface has max of 8 materials, each may require two
    // (inside/outside medium) + every (light + camera) surface
    // having unique medium (Utilizing arbitrary count of 512)
    // Worst case, we will have couple of rehashes nothing critical.
    ItemLocationMap mediumHT;
    mediumHT.reserve(512);
    std::future<void> mediumHTReady = threadPool.submit_task(
    [&mediumHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        return CreateHT(mediumHT, sceneJson.at(NodeNames::MEDIUM_LIST));
    });
    // Textures
    // It is hard to find estimate the worst case texture count as well.
    // Simple Heuristic: Each surface has unique material, each requiring
    // two textures, there are total of 16 mediums each require a single
    // texture
    ItemLocationMap textureHT;
    textureHT.reserve(surfaces.size() * MaxPrimBatchPerSurface * 2 +
                      16);
    std::future<void> textureHTReady = threadPool.submit_task(
    [&textureHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        return CreateHT(textureHT, sceneJson.at(NodeNames::TEXTURE_LIST));
    });

    // Check boundary first
    auto PushToTypeMapping =
    [&sceneJson = std::as_const(sceneJson)](TypeMappedNodes& typeMappings,
                                            const ItemLocationMap& map, uint32_t id,
                                            const std::string_view& listName,
                                            bool skipUnknown = false)
    {
        const auto& it = map.find(id);
        if(skipUnknown && it == map.end()) return;

        if(it == map.end())
            throw MRayError("Id({:d}) could not be "
                            "located in \"{:s}\"",
                            id, listName);
        const auto& location =  it->second;
        uint32_t arrayIndex = std::get<ARRAY_INDEX>(location);
        uint32_t innerIndex = std::get<INNER_INDEX>(location);

        auto node = JsonNode(sceneJson[listName][arrayIndex], innerIndex);
        std::string_view type = node.Type();
        auto result = typeMappings[type].emplace_back(std::move(node));
    };

    // Start with boundary
    using namespace NodeNames;
    lightHTReady.wait();
    PushToTypeMapping(lightNodes, lightHT, boundary.lightId, LIGHT_LIST);
    mediumHTReady.wait();
    PushToTypeMapping(mediumNodes, mediumHT, boundary.mediumId, MEDIUM_LIST);
    transformHTReady.wait();
    PushToTypeMapping(transformNodes, transformHT, boundary.transformId, TRANSFORM_LIST);

    // Prim/Material Surfaces
    matHTReady.wait();
    primHTReady.wait();

    std::vector<NodeTexStruct> textureIds;
    textureIds.reserve(surfaces.size() * 2);
    for(const auto& s : surfaces)
    {
        for(uint8_t i = 0; i < s.pairCount; i++)
        {
            uint32_t matId = std::get<SurfaceStruct::MATERIAL_INDEX>(s.matPrimBatchPairs[i]);
            uint32_t primId = std::get<SurfaceStruct::PRIM_INDEX>(s.matPrimBatchPairs[i]);
            PushToTypeMapping(materialNodes, matHT, matId, MATERIAL_LIST);
            PushToTypeMapping(primNodes, primHT, primId, PRIMITIVE_LIST);
            PushToTypeMapping(transformNodes, transformHT,
                              s.transformId, TRANSFORM_LIST);

            if(s.alphaMaps[i].has_value())
                textureIds.push_back(s.alphaMaps[i].value());
        }
    }
    // Camera Surfaces
    camHTReady.wait();
    for(const auto& c : camSurfaces)
    {
        PushToTypeMapping(cameraNodes, camHT, c.cameraId, CAMERA_LIST);
        PushToTypeMapping(mediumNodes, mediumHT, c.mediumId, MEDIUM_LIST);
        PushToTypeMapping(transformNodes, transformHT,
                          c.transformId, TRANSFORM_LIST, true);
    }
    // Light Surfaces
    lightHTReady.wait();
    for(const auto& l : lightSurfaces)
    {
        l.lightId;
        PushToTypeMapping(lightNodes, lightHT, l.lightId, LIGHT_LIST);
        PushToTypeMapping(mediumNodes, mediumHT, l.mediumId, MEDIUM_LIST);
        PushToTypeMapping(transformNodes, transformHT,
                          l.transformId, TRANSFORM_LIST, true);
    }

    // Now double indirections...
    // We need to iterate lights once to find primitive's that are required
    // by these lights. Lights are generic so we need look at the light via
    // the tracer. We call this dry run, since we do most of the scene-related
    // load work but we do not actually load the data.
    std::vector<uint32_t> primitiveIds;
    primitiveIds.reserve(lightSurfaces.size());
    DryRunLightsForPrim(primitiveIds, lightNodes, tracer);
    for(const auto& p : primitiveIds)
        PushToTypeMapping(primNodes, primHT, p, PRIMITIVE_LIST);

    // This is true for textures as well. Materials may/may not require textures
    // (or mediums) so we need to check these as well
    DryRunNodesForTex(textureIds, materialNodes, tracer,
                      &TracerI::MatAttributeInfo);
    DryRunNodesForTex(textureIds, mediumNodes, tracer,
                      &TracerI::MediumAttributeInfo);

    // And finally create texture mappings
    textureHTReady.wait();
    for(const auto& t : textureIds)
    {
        const auto& it = textureHT.find(t.texId);
        if(it == textureHT.end())
            throw MRayError("Id({:d}) could not be "
                            "located in {:s}",
                            t.texId, TEXTURE_LIST);
        const auto& location = it->second;
        uint32_t arrayIndex = std::get<ARRAY_INDEX>(location);
        uint32_t innerIndex = std::get<INNER_INDEX>(location);
        bool isMultiNode = std::get<IS_MULTI_NODE>(location);
        auto node = JsonNode(sceneJson[TEXTURE_LIST][arrayIndex], innerIndex);
        textureNodes.emplace(t, std::move(node));
    }
}

void SceneLoaderMRay::CreateSurfaces(TracerI&, const std::vector<SurfaceStruct>&)
{

}

void SceneLoaderMRay::CreateLightSurfaces(TracerI&, const std::vector<LightSurfaceStruct>&)
{

}

void SceneLoaderMRay::CreateCamSurfaces(TracerI&, const std::vector<CameraSurfaceStruct>&)
{

}

MRayError SceneLoaderMRay::LoadAll(TracerI& tracer)
{
    using Node = Optional<const nlohmann::json*>;
    auto FindNode = [this](std::string_view str) -> Node
    {
        const auto i = sceneJson.find(str);
        if(i == sceneJson.end()) return std::nullopt;
        return &(*i);
    };
    using namespace NodeNames;

    const nlohmann::json emptyJson;
    Node camSurfJson    = FindNode(CAMERA_SURFACE_LIST);
    Node lightSurfJson  = FindNode(LIGHT_SURFACE_LIST);
    Node surfJson       = FindNode(SURFACE_LIST);
    if(!camSurfJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" array", CAMERA_SURFACE_LIST);
    if(!lightSurfJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" array", LIGHT_SURFACE_LIST);
    if(!surfJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" array", SURFACE_LIST);
    // Check the boundary light
    Node boundaryJson = FindNode(BOUNDARY);
    if(!boundaryJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" object", BOUNDARY);

    // Now many things may fail (wrong name's, wrong types etc)
    // go full exceptions here (utilize the json as much as possible)
    // TODO: Change this to std::expected maybe c++23?
    try
    {
        LightSurfaceStruct boundary = LoadBoundary(*boundaryJson.value());
        SceneSurfList surfaces = LoadSurfaces(*surfJson.value());
        SceneCamSurfList camSurfs = LoadCamSurfaces(*camSurfJson.value(),
                                                    boundary.mediumId);
        SceneLightSurfList lightSurfs = LoadLightSurfaces(*lightSurfJson.value(),
                                                          boundary.mediumId);
        // Surfaces are loaded now create type/ node pairings
        // These are stored in the loader's state
        CreateTypeMapping(tracer, surfaces, camSurfs,
                          lightSurfs, boundary);

        // Multi-threaded section
        ExceptionList exceptionList;

        // Many things depend on textures, so this is first
        // (currently only materials/alpha mapped surfaces,
        // and mediums)
        LoadTextures(tracer, exceptionList);
        // Technically, we should not wait here, only materials
        // and surfaces depend on textures.
        // We are waiting here for future proofness, in future
        // or a user may create a custom primitive type that holds
        // a texture etc.
        threadPool.wait();
        // Types that depend on textures
        LoadMediums(tracer, exceptionList);
        // Waiting here because Materials depend on mediums
        // In mray, materials seperate two mediums.
        threadPool.wait();

        LoadMaterials(tracer, exceptionList);
        // Does not depend on textures but may depend on later
        LoadTransforms(tracer, exceptionList);
        LoadPrimitives(tracer, exceptionList);
        LoadCameras(tracer, exceptionList);
        // Lights may depend on primitives (primitive-backed lights)
        // So we need to wait primitive id mappings to complete
        threadPool.wait();
        LoadLights(tracer, exceptionList);

        // Finally, wait all load operations to complete
        threadPool.wait();

        // Check if any exceptions are occured
        // Concat to a single exception and return it
        if(exceptionList.size != 0)
        {
            MRayError err("");
            for(const auto& e : exceptionList.exceptions)
            {
                err.AppendInfo(e.GetError() + "\n");
            }
            return err;
        }

        // Scene id -> tracer id mappings are created
        // and reside on the object's state.
        // Issue surface mappings to the tracer
        // Back to single thread here, a scene (even large)
        // probably consists of mid-thousands of surfaces.
        // Also this has a single bottleneck unlike tracer groups,
        // so it probably not worth it.
        CreateSurfaces(tracer, surfaces);
        CreateLightSurfaces(tracer, lightSurfs);
        CreateCamSurfaces(tracer, camSurfs);

        // Wait the surface creations
        threadPool.wait();
    }
    // MRay related errros
    catch(const MRayError& e)
    {
        threadPool.purge();
        threadPool.wait();
        return e;
    }
    // Json related errors
    catch(const nlohmann::json::exception& e)
    {
        threadPool.purge();
        threadPool.wait();
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

MRayError SceneLoaderMRay::OpenFile(const std::string& filePath)
{
    const auto path = std::filesystem::path(filePath);
    std::ifstream file(path);

    if(!file.is_open())
        return MRayError("Scene file \"{}\" is not found",
                         filePath);
    // Parse Json
    try
    {
        sceneJson = nlohmann::json::parse(file, nullptr, true, true);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

MRayError SceneLoaderMRay::ReadStream(std::istream& sceneData)
{
    // Parse Json
    try
    {
        sceneJson = nlohmann::json::parse(sceneData, nullptr, true, true);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

SceneLoaderMRay::SceneLoaderMRay(BS::thread_pool& pool)
    :threadPool(pool)
{}

Pair<MRayError, double> SceneLoaderMRay::LoadScene(TracerI& tracer,
                                                   const std::string& filePath)
{
    Timer t; t.Start();
    MRayError e = MRayError::OK;
    if(e = OpenFile(filePath)) return {e, -0.0};
    if(e = LoadAll(tracer)) return {e, -0.0};
    t.Split();

    return {MRayError::OK, t.Elapsed<Second>()};
}

Pair<MRayError, double> SceneLoaderMRay::LoadScene(TracerI& tracer,
                                                   std::istream& sceneData)
{
    Timer t; t.Start();
    MRayError e = MRayError::OK;
    if(e = ReadStream(sceneData)) return {e, -0.0};
    if(e = LoadAll(tracer)) return {e, -0.0};
    t.Split();

    return {MRayError::OK, t.Elapsed<Second>()};
}