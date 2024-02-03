#include "SceneLoaderMRay.h"

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "Core/Timer.h"

#include "MeshLoader/EntryPoint.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>

#include <barrier>
#include <atomic>

#include "JsonNode.h"

static void LoadPrimitive(TracerI& tracer,
                          PrimGroupId groupId,
                          PrimBatchId batchId,
                          uint32_t meshInternalIndex,
                          const std::unique_ptr<MeshFileI>& meshFile)
{
    using enum PrimitiveAttributeLogic;
    const auto& attributeList = tracer.PrimAttributeInfo(groupId);

    for(uint32_t attributeIndex = 0;
        attributeIndex < attributeList.size();
        attributeIndex++)
    {
        const auto& attribute = attributeList[attributeIndex];
        PrimitiveAttributeLogic attribLogic = std::get<0>(attribute);
        AttributeOptionality optionality = std::get<1>(attribute);
        MRayDataTypeRT groupsLayout = std::get<2>(attribute);
        MRayDataTypeRT filesLayout = meshFile->AttributeLayout(attribLogic,
                                                               meshInternalIndex);

        // Is this data available?
        if(!meshFile->HasAttribute(attribLogic))
        {
            if(optionality == AttributeOptionality::MR_MANDATORY)
                throw MRayError(MRAY_FORMAT("Mesh File{:s}:[{:d}] do not have \"{}\""
                                            "which is mandatory for {}",
                                            meshFile->Name(), meshInternalIndex,
                                            PrimAttributeStringifier::ToString(attribLogic),
                                            tracer.TypeName(groupId)));
            continue;
        }
        // Special case for default triangle
        if(attribLogic == NORMAL && groupsLayout.type == MRayDataEnum::MR_QUATERNION &&
           meshFile->HasAttribute(TANGENT) && meshFile->HasAttribute(BITANGENT) &&
           meshFile->AttributeLayout(TANGENT).type == MRayDataEnum::MR_VECTOR_3 &&
           meshFile->AttributeLayout(BITANGENT).type == MRayDataEnum::MR_VECTOR_3 &&
           meshFile->AttributeLayout(NORMAL).type == MRayDataEnum::MR_VECTOR_3)
        {
            size_t normalCount = meshFile->MeshAttributeCount(meshInternalIndex);
            MRayInput q(std::in_place_type_t<Quaternion>{}, normalCount);
            Span<Quaternion> quats = q.AccessAs<Quaternion>();

            // Utilize TBN matrix directly
            MRayInput t = meshFile->GetAttribute(TANGENT, meshInternalIndex);
            MRayInput b = meshFile->GetAttribute(BITANGENT, meshInternalIndex);
            MRayInput n = meshFile->GetAttribute(attribLogic, meshInternalIndex);

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
        else if(groupsLayout.type != filesLayout.type)
        {
            // TODO: Add CPU conversion logic here for basic conversions
            // (Both is floating point type and conversion is *not* narrowing)
            // For the rest below error should be fine

            // We require exact match currently
            throw MRayError(MRAY_FORMAT("Mesh File{:s}:[{:d}]'s data layout of \"{}\""
                                        "(has type{:s}) does not match the {}'s data layout "
                                        "(which is {:s})",
                                        meshFile->Name(), meshInternalIndex,
                                        PrimAttributeStringifier::ToString(attribLogic),
                                        MRayDataTypeStringifier::ToString(filesLayout.type),
                                        tracer.TypeName(groupId),
                                        MRayDataTypeStringifier::ToString(groupsLayout.type)));
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

void SceneLoaderMRay::CreateTypeMapping(const std::vector<SurfaceStruct>&,
                                        const std::vector<CameraSurfaceStruct>&,
                                        const std::vector<LightSurfaceStruct>&)
{

}

void SceneLoaderMRay::LoadTextures(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::LoadMediums(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::LoadMaterials(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::LoadTransforms(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::LoadPrimitives(TracerI& tracer, ExceptionList& exceptions)
{
    // Issue loads to the thread pool
    for(const auto& m : primNodes)
    {
        struct PrimGroupBatchList
        {
            std::atomic_size_t allocator = 0;
            std::vector<Pair<uint32_t, PrimBatchId>> flatGroupBatches;
        };
        auto groupBatchList = std::make_shared<PrimGroupBatchList>();

        std::string_view primTypeName = m.first;
        PrimGroupId groupId = tracer.CreatePrimitiveGroup(std::string(primTypeName));

        // Construct Barrier
        auto BarrierFunc = [groupId, groupBatchList, &tracer]() noexcept
        {
            // When barrier completed
            // Reserve the space for mappings
            // Commit prim greoups reservations
            tracer.CommitPrimReservations(groupId);
            groupBatchList->flatGroupBatches.resize(groupBatchList->allocator);
            groupBatchList->allocator = 0;
        };
        auto barrier = std::make_shared<std::barrier<decltype(BarrierFunc)>>(threadPool.get_thread_count(),
                                                                             BarrierFunc);
        // Determine the thread size
        uint32_t threadCount = std::min(threadPool.get_thread_count(),
                                        static_cast<uint32_t>(m.second.size()));
        auto future = threadPool.submit_blocks(std::size_t(0), m.second.size(),
        // Copy the shared pointers, capture by reference the rest
        [&, this, barrier, groupBatchList](size_t start, size_t end)
        {
            std::vector<Pair<uint32_t, PrimBatchId>> localBatchList;
            std::map<std::string, std::unique_ptr<MeshLoaderI>> loaders;
            std::map<std::string, std::unique_ptr<MeshFileI>> meshFiles;
            std::vector<Pair<uint32_t, const std::unique_ptr<MeshFileI>*>> batchFiles;

            batchFiles.reserve(end - start);
            try
            {
                for(size_t i = start; i < end; i++)
                {
                    const MRayJsonNode& node = m.second[i];
                    std::string_view fileName = node.CommonData<std::string_view>(NodeNames::NAME);
                    uint32_t innerIndex = node.AccessData<uint32_t>(NodeNames::INNER_INDEX);

                    std::string key = std::filesystem::path(fileName).stem().string();
                    if(fileName != NodeNames::NODE_PRIMITIVE &&
                       fileName != NodeNames::INDEXED_NODE_PRIMITIVE)
                    {
                        std::string_view tag = node.Tag();
                        key += tag;
                    }

                    const auto& loaderIt = loaders.find(key);
                    const auto& ml = (loaderIt != loaders.end())
                                        ? loaderIt->second
                                        : loaders.emplace(key,
                                                          meshLoaderPool->AcquireALoader(key)).first->second;

                    std::string fileAbsolutePath = SceneLoaderMRay::SceneRelativePathToAbsolute(fileName, scenePath);
                    const auto& fileIt = meshFiles.find(fileAbsolutePath);
                    const auto& meshFile = (fileIt != meshFiles.end())
                                            ? fileIt->second
                                            : meshFiles.emplace(fileAbsolutePath,
                                                                ml->OpenFile(fileAbsolutePath)).first->second;

                    PrimCount pc
                    {
                        .primCount = meshFile->MeshPrimitiveCount(innerIndex),
                        .attributeCount = meshFile->MeshAttributeCount(innerIndex)
                    };
                    PrimBatchId batchId = tracer.ReservePrimitiveBatch(groupId, pc);
                    batchFiles[i - start] = Pair(innerIndex, &meshFile);
                    groupBatchList->allocator++;
                    localBatchList[i - start] = Pair(node.Id(), batchId);
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
                    const auto& batchFile = batchFiles[i - start];

                    // First, write the batch ids, to the list
                    size_t location = groupBatchList->allocator.fetch_add(localBatchList.size());
                    std::copy(localBatchList.cbegin(), localBatchList.cend(),
                              groupBatchList->flatGroupBatches.begin() + location);
                    // Load
                    const auto& batchId = localBatchList[i - start].second;
                    LoadPrimitive(tracer, groupId, batchId,
                                  batchFile.first, *batchFile.second);
                }
            }
            catch(MRayError& e)
            {
                exceptions.AddException(std::move(e));
            }
        }, threadCount);


        // Move future to shared_ptr
        using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
        auto futureShared = FutureSharedPtr(new BS::multi_future<void>(std::move(future)));

        // Issue a one final task that pushes the primitives to the global map
        threadPool.detach_task([&, this, future = futureShared, groupBatchList]()
        {
            // Wait other tasks to complere
            future->wait();
            // After this point groupBatchList is fully loaded
            std::scoped_lock lock(primMappings.mutex);
            primMappings.map.insert(groupBatchList->flatGroupBatches.cbegin(),
                                groupBatchList->flatGroupBatches.cend());
        });
    }
}

void SceneLoaderMRay::LoadCameras(TracerI&, ExceptionList&)
{

}

void SceneLoaderMRay::LoadLights(TracerI&, ExceptionList&)
{

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
        return MRayError(MRAY_FORMAT("Scene file does not contain "
                                     "\"{}\" array", CAMERA_SURFACE_LIST));
    if(!lightSurfJson.has_value())
        return MRayError(MRAY_FORMAT("Scene file does not contain "
                                     "\"{}\" array", LIGHT_SURFACE_LIST));
    if(!surfJson.has_value())
        return MRayError(MRAY_FORMAT("Scene file does not contain "
                                     "\"{}\" array", SURFACE_LIST));

    // Check the boundary light
    Node boundaryJson = FindNode(BOUNDARY);
    if(!boundaryJson.has_value())
        return MRayError(MRAY_FORMAT("Scene file does not contain "
                                     "\"{}\" object", BOUNDARY));

    // Now many things may fail (wrong name's, wrong types etc)
    // go full exceptions here (utilize the json as much as possible)
    // TODO: Change this to std::expected maybe c++23?
    try
    {
        LightSurfaceStruct boundary                 = LoadBoundary(*boundaryJson.value());
        std::vector<SurfaceStruct> surfaces         = LoadSurfaces(*camSurfJson.value());
        std::vector<CameraSurfaceStruct> camSurfs   = LoadCamSurfaces(*lightSurfJson.value(),
                                                                      boundary.mediumId);
        std::vector<LightSurfaceStruct> lightSurfs  = LoadLightSurfaces(*surfJson.value(),
                                                                        boundary.mediumId);

        // Surfaces are loaded now create type/ node pairings
        // These are stored in the loader's state
        CreateTypeMapping(surfaces, camSurfs, lightSurfs);

        // Multi-threaded section
        ExceptionList exceptionList;

        // Many things depend on textures, so this is first
        // (currently only materials/alpha mapped surfaces,
        // and mediums)
        LoadTextures(tracer, exceptionList);
        // Technically, we should not wait here only materials
        // and surfaces depend on textures.
        // We are waiting here for future proofness, in future
        // or a user may create a custom primitive type that holds
        // a texture etc.
        threadPool.wait();
        // Types that depend on textures
        LoadMediums(tracer, exceptionList);
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
        if(exceptionList.exceptions.size() != 0)
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
        return e;
    }
    // Json related errors
    catch(const nlohmann::json::exception& e)
    {
        return MRayError(std::string(e.what()));
    }
    return MRayError::OK;
}

MRayError SceneLoaderMRay::OpenFile(const std::string& filePath)
{
    const auto path = std::filesystem::path(filePath);
    std::ifstream file(path);

    if(!file.is_open())
        return MRayError(MRAY_FORMAT("Scene file \"{}\" is not found",
                                     filePath));
    // Parse Json
    try
    {
        sceneJson = nlohmann::json::parse(file, nullptr, true, true);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return MRayError(std::string(e.what()));
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