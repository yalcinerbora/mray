#include "GFGConverter.h"

#include <filesystem>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/DefaultLogger.hpp>

#include <BS/BS_thread_pool.hpp>

#include <gfg/GFGFileExporter.h>

#include <nlohmann/json.hpp>

#include "Core/Timer.h"
#include "Core/MemAlloc.h"
#include "Core/Vector.h"
#include "Core/Quaternion.h"
#include "Core/Filesystem.h"

// TODO: Type leak change it? (This functionaly is
// highly intrusive, probably needs a redesign?)
#include "SceneLoaderMRay/NodeNames.h"

static const std::string ASSIMP_TAG = "assimp";
static const std::string GFG_TAG = "gfg";

struct MeshGroup
{
    std::vector<uint32_t>   primIds;
    std::vector<uint32_t>   innerIndices;
    std::string             filePath;
    std::string             tag;
};

Expected<nlohmann::json> OpenFile(const std::string& filePath)
{
    std::ifstream file(filePath);
    if(!file.is_open())
        return MRayError("Scene file \"{}\" is not found",
                         filePath);
    // Parse Json
    try
    {
        return nlohmann::json::parse(file, nullptr, true, true);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
}

// Simple wrapper to utilize "MultiData" allocation scheme
struct VectorBackedMemory
{
    std::vector<uint8_t> v;

    void ResizeBuffer(size_t s) { v.resize(s); }
    size_t Size() const { return v.size(); }

    explicit operator const Byte*() const { return reinterpret_cast<const Byte*>(v.data()); }
    explicit operator Byte* () { return reinterpret_cast<Byte*>(v.data()); }
};
static_assert(MemoryC<VectorBackedMemory>);

Expected<std::vector<MeshGroup>> FindMeshes(const nlohmann::json& sceneJson)
{
    if(auto primNodes = sceneJson.find(NodeNames::PRIMITIVE_LIST);
       primNodes != sceneJson.end())
    {
        // Arbitrarly reserve some, to prevent many early mallocs
        std::vector<MeshGroup> parsedMeshes;
        parsedMeshes.reserve(512);
        for(const auto& primNode : *primNodes)
        {
            const auto& tag = primNode.at(NodeNames::TAG).get<std::string_view>();
            const auto& idList = primNode.at(NodeNames::ID);

            // Skip in node triangles, spheres
            if(tag == NodeNames::NODE_PRIM_TRI ||
               tag == NodeNames::NODE_PRIM_TRI_INDEXED ||
               tag == NodeNames::NODE_PRIM_SPHERE)
                continue;

            const auto& innerIndexList = primNode.at(NodeNames::INNER_INDEX);
            if(tag != ASSIMP_TAG)
                return MRayError("All mesh files must be "
                                 "tagged with \"assimp\"!");

            MeshGroup group;
            group.filePath = primNode.at(NodeNames::FILE).get<std::string>();
            if(idList.is_array())
            {
                group.primIds = idList.get<std::vector<uint32_t>>();
                group.innerIndices = innerIndexList.get<std::vector<uint32_t>>();

            }
            else
            {
                group.primIds.push_back(idList.get<uint32_t>());
                group.innerIndices.push_back(innerIndexList.get<uint32_t>());
            }
            parsedMeshes.push_back(std::move(group));
        }
        return Expected<std::vector<MeshGroup>>(std::move(parsedMeshes));
    }
    return MRayError("There are no meshes to convert!");
}

MRayError THRDProcessMesh(Span<const MeshGroup> meshes,

                          GFGFileExporter& exporter,
                          std::mutex& exporterMutex,

                          const std::string& inScenePath,
                          MRayConvert::ConversionFlags flags)
{
    using enum MRayConvert::ConvFlagEnum;

    // Get assimp importer, all meshes assumed to be loaded via
    // assimp.
    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE,
                                aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    // Import only position, normal, tangent (if provided) and uv
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                aiComponent_BONEWEIGHTS |
                                aiComponent_COLORS);
    static constexpr unsigned int assimpFlags = static_cast<unsigned int>
    (
        // Generate Bounding Boxes
        aiProcess_GenBoundingBoxes |
        // Generate Normals if not avail
        aiProcess_GenNormals |
        // Generate Tangent and Bi-tangents if not avail
        aiProcess_CalcTangentSpace |
        // Triangulate
        aiProcess_Triangulate |
        // Improve triangle order
        aiProcess_ImproveCacheLocality |
        // Reduce Vertex Count
        aiProcess_JoinIdenticalVertices |
        // Remove Degenerate triangles
        aiProcess_FindDegenerates |
        // Sort by primitive type
        // (this guarantees a "mesh" has same types triangles)
        aiProcess_SortByPType |
        //
        aiProcess_RemoveRedundantMaterials
    );


    // Thread local buffers, used to
    VectorBackedMemory indexMem;
    VectorBackedMemory meshMem;

    for(const auto& mesh : meshes)
    {
        std::string meshPath = Filesystem::RelativePathToAbsolute(mesh.filePath,
                                                                  inScenePath);

        // TODO: Delete previously created files??
        const aiScene* assimpScene = importer.ReadFile(meshPath, assimpFlags);
        if(!assimpScene) return MRayError("Assimp: Unable to read file \"{}\"",
                                          meshPath);

        // GFG require user to lay the data
        // We will push the data as SoA style
        for(uint32_t meshIndex : mesh.innerIndices)
        {
            const auto& meshIn = assimpScene->mMeshes[meshIndex];
            uint32_t vertexCount = meshIn->mNumVertices;
            uint32_t indexCount = meshIn->mNumFaces * 3;

            assert(meshIn->HasPositions());
            assert(meshIn->HasNormals());
            assert(meshIn->HasTangentsAndBitangents());

            GFGMeshHeaderCore header = {};
            header.aabb.max[0] = meshIn->mAABB.mMax[0];
            header.aabb.max[1] = meshIn->mAABB.mMax[1];
            header.aabb.max[2] = meshIn->mAABB.mMax[2];
            header.aabb.min[0] = meshIn->mAABB.mMin[0];
            header.aabb.min[1] = meshIn->mAABB.mMin[1];
            header.aabb.min[2] = meshIn->mAABB.mMin[2];

            header.componentCount = (flags[NORMAL_AS_QUATERNION])
                                        ? 3 : 5;

            header.indexCount = indexCount;
            header.indexSize = sizeof(uint32_t);
            header.topology = GFGTopology::TRIANGLE;
            header.vertexCount = vertexCount;

            std::vector<GFGVertexComponent> components;
            components.reserve(header.componentCount);
            size_t offset = 0;
            components.push_back(GFGVertexComponent
                                 {
                                     .dataType = GFGDataType::FLOAT_3,
                                     .logic = GFGVertexComponentLogic::POSITION,
                                     .startOffset = offset,
                                     .internalOffset = 0,
                                     .stride = sizeof(float) * 3

                                 });
            offset += vertexCount * components.back().stride;
            components.push_back(GFGVertexComponent
                                 {
                                     .dataType = GFGDataType::FLOAT_2,
                                     .logic = GFGVertexComponentLogic::UV,
                                     .startOffset = offset,
                                     .internalOffset = 0,
                                     .stride = sizeof(float) * 2
                                 });
            offset += vertexCount * components.back().stride;
            if(flags[NORMAL_AS_QUATERNION])
            {
                components.push_back(GFGVertexComponent
                                     {
                                         .dataType = GFGDataType::QUATERNION,
                                         .logic = GFGVertexComponentLogic::NORMAL,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 4
                                     });
                offset += vertexCount * components.back().stride;
            }
            else
            {
                components.push_back(GFGVertexComponent
                                     {
                                         .dataType = GFGDataType::FLOAT_3,
                                         .logic = GFGVertexComponentLogic::NORMAL,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 3

                                     });
                offset += vertexCount * components.back().stride;
                components.push_back(GFGVertexComponent
                                     {
                                         .dataType = GFGDataType::FLOAT_3,
                                         .logic = GFGVertexComponentLogic::TANGENT,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 3

                                     });
                offset += vertexCount * components.back().stride;
                components.push_back(GFGVertexComponent
                                     {
                                         .dataType = GFGDataType::FLOAT_3,
                                         .logic = GFGVertexComponentLogic::BINORMAL,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 3

                                     });
                offset += vertexCount * components.back().stride;
            }

            Span<Vector3ui> indices;
            MemAlloc::AllocateMultiData(std::tie(indices), indexMem,
                                        {meshIn->mNumFaces}, 1u);


            Span<Vector3> positions;
            Span<Vector2> uvs;
            Span<Quaternion> normalsQuat;
            Span<Vector3> normals;
            Span<Vector3> tangents;
            Span<Vector3> bitangents;

            if(flags[NORMAL_AS_QUATERNION])
            {
                size_t alignment = std::max(std::max(sizeof(Vector3),
                                                     sizeof(Vector2)),
                                            sizeof(Quaternion));
                MemAlloc::AllocateMultiData(std::tie(positions, uvs,
                                                     normalsQuat),
                                            meshMem,
                                            {vertexCount, vertexCount,
                                            vertexCount},
                                            alignment);
            }
            else
            {
                size_t alignment = std::max(sizeof(Vector3),
                                            sizeof(Vector2));
                                            MemAlloc::AllocateMultiData(std::tie(positions, uvs,
                                                                                 normals, tangents, bitangents),
                                                                        meshMem,
                                                                        {vertexCount, vertexCount,
                                                                        vertexCount, vertexCount,
                                                                        vertexCount},
                                                                        alignment);
            }


            // Indices
            for(unsigned int i = 0; i < meshIn->mNumFaces; i++)
            {

                const auto& face = meshIn->mFaces[i];
                Vector3ui faceIndices(face.mIndices[0],
                                      face.mIndices[1],
                                      face.mIndices[2]);
                indices[i] = faceIndices;
            }
            // Positions
            static_assert(sizeof(Vector3) == sizeof(aiVector3D));
            std::memcpy(positions.data(), meshIn->mVertices,
                        vertexCount * sizeof(Vector3));
            // UVs
            static_assert(sizeof(Vector2) == sizeof(aiVector2D));
            // Manual copy due to stride
            if(!meshIn->HasTextureCoords(0))
            {
                std::memset(uvs.data(), 0x00, uvs.size_bytes());
            }
            else for(unsigned int i = 0; i < meshIn->mNumVertices; i++)
            {
                uvs[i] = Vector2(meshIn->mTextureCoords[0][i].x,
                                 meshIn->mTextureCoords[0][i].y);
            }
            // Normals
            if(flags[NORMAL_AS_QUATERNION])
            {
                for(unsigned int i = 0; i < meshIn->mNumVertices; i++)
                {
                    Vector3 t(meshIn->mTangents[i].x,
                              meshIn->mTangents[i].y,
                              meshIn->mTangents[i].z);
                    Vector3 b(meshIn->mBitangents[i].x,
                              meshIn->mBitangents[i].y,
                              meshIn->mBitangents[i].z);
                    Vector3 n(meshIn->mNormals[i].x,
                              meshIn->mNormals[i].y,
                              meshIn->mNormals[i].z);
                    Quaternion quat = TransformGen::ToSpaceQuat(t, b, n);
                    normalsQuat[i] = quat;
                }
            }
            else
            {
                std::memcpy(normals.data(), meshIn->mNormals,
                            vertexCount * sizeof(Vector3));
                std::memcpy(tangents.data(), meshIn->mTangents,
                            vertexCount * sizeof(Vector3));
                std::memcpy(bitangents.data(), meshIn->mBitangents,
                            vertexCount * sizeof(Vector3));
            }


            // Laid out the data now add to the mesh
            // If global gfg is used, lock it.
            // If is local just add
            if(flags[PACK_GFG])
            {
                // If packed
                std::scoped_lock<std::mutex> lock(exporterMutex);
                exporter.AddMesh(0, components,
                                 header, meshMem.v,
                                 &(indexMem.v));
            }
            else
            {
                exporter.AddMesh(0, components,
                                 header, meshMem.v,
                                 &(indexMem.v));
            }

        }

        // If not packed, locally write the item.
        if(!flags[PACK_GFG])
        {
            namespace fs = std::filesystem;
            auto gfgPath = fs::path(meshPath).replace_extension(fs::path(GFG_TAG));
            fs::create_directories(fs::absolute(gfgPath).remove_filename());
            std::ofstream outFile(gfgPath, std::ofstream::binary);
            GFGFileWriterSTL fileWriter(outFile);
            exporter.Write(fileWriter);
        }
    }

    return MRayError::OK;
}

MRayError ValidateOutputFiles(MRayConvert::ConversionFlags flags,
                              const std::string& inFileName,
                              const std::string& outFileName,
                              const std::vector<MeshGroup>& parsedMeshes)
{
    using enum MRayConvert::ConvFlagEnum;
    namespace fs = std::filesystem;

    std::string inScenePath = fs::path(inFileName).generic_string();

    if(flags[FAIL_ON_OVERWRITE])
    {
        if(flags[PACK_GFG])
        {
            auto outGFGName = fs::path(inFileName).filename();
            outGFGName.replace_extension(fs::path(GFG_TAG));
            auto outGFGPath = (fs::path(outFileName).remove_filename() /
                               outFileName);

            if(fs::exists(outGFGPath))
                return MRayError("GFG file \"{}\" already exists",
                                 outGFGPath.generic_string());
        }
        else for(const auto& mesh : parsedMeshes)
        {

            std::string meshPath = Filesystem::RelativePathToAbsolute(mesh.filePath,
                                                                      inScenePath);
            auto gfgPath = fs::path(meshPath).replace_extension(fs::path(GFG_TAG));

            if(fs::exists(gfgPath))
                return MRayError("GFG file \"{}\" already exists",
                                 gfgPath.generic_string());
        }
    }
    return MRayError::OK;
}

MRAY_GFGCONVERTER_ENTRYPOINT
Expected<double> MRayConvert::ConvertMeshesToGFG(const std::string& outFileName,
                                                 const std::string& inFileName,
                                                 ConversionFlags flags)
{
    BS::thread_pool threadPool;


    using enum ConvFlagEnum;
    namespace fs = std::filesystem;
    if(flags[FAIL_ON_OVERWRITE] &&
       fs::exists(fs::path(outFileName)))
    {
        return MRayError("\"{}\" already exists!",
                         outFileName);
    }

    std::string inScenePath = fs::path(inFileName).remove_filename().generic_string();

    // Do the loading
    Timer timer; timer.Start();
    Expected<nlohmann::json> sceneJsonE = OpenFile(inFileName);
    if(sceneJsonE.has_error())
        return sceneJsonE.error();
    const auto& sceneJson = sceneJsonE.value();

    try
    {

        // Find the meshes
        Expected<std::vector<MeshGroup>> meshOut = FindMeshes(sceneJson);
        if(meshOut.has_error())
            return meshOut.error();
        const auto& parsedMeshes = meshOut.value();

        // Do file availability validation
        // If don't override flag is set this will give an error
        auto e = ValidateOutputFiles(flags, inFileName, outFileName,
                                     parsedMeshes);
        if(e) return e;


        // Multi-threaded process meshes
        // Normally this wont give much perf, but we do postprocess
        // meshes via assimp to create tangents etc. So this has
        // some gain.
        Span<const MeshGroup> meshes(parsedMeshes);
        std::mutex exporterMutex;
        GFGFileExporter globalExporter;
        auto future = threadPool.submit_blocks(static_cast<size_t>(0),
                                               parsedMeshes.size(),
                                               [&, meshes, flags](size_t start,
                                                                  size_t end)
        {
            auto mySpan = meshes.subspan(start, end - start);
            GFGFileExporter localExporter;
            GFGFileExporter* expRef = (flags[PACK_GFG]) ? &globalExporter
                                                        : &localExporter;
            return THRDProcessMesh(mySpan, *expRef, exporterMutex,
                                   inScenePath, flags);
        });

        // Process the errors if any
        auto errors = future.get();
        MRayError errOut = MRayError::OK;
        bool hasErrors = false;
        for(MRayError& err : errors)
        {
            hasErrors = hasErrors || err;
            if(err)
                errOut.AppendInfo(err.GetError() + "\n");
        }
        if(hasErrors) return errOut;


        // Either all the meshes are processed and written on disk
        // or we are packing all the meshes on a single GFG
        // we need to write it from the main thread.
        if(flags[PACK_GFG])
        {
            auto gfgPath = fs::path(outFileName).replace_extension(fs::path(GFG_TAG));
            fs::create_directories(fs::absolute(gfgPath).remove_filename());
            std::ofstream outFile(gfgPath, std::ofstream::binary);
            GFGFileWriterSTL fileWriter(outFile);
            globalExporter.Write(fileWriter);
            globalExporter.Clear();
        }

        // Write complete
        // Now we need to convert the json file
        // Copy the file representation
        nlohmann::json outJson = sceneJson;
        // Add in node primitives
        nlohmann::json newPrimArray;
        for(const auto& primNode : sceneJson.at(NodeNames::PRIMITIVE_LIST))
        {
            const auto& tag = primNode.at(NodeNames::TAG).get<std::string_view>();
            if(tag == NodeNames::NODE_PRIM_TRI ||
               tag == NodeNames::NODE_PRIM_TRI_INDEXED ||
               tag == NodeNames::NODE_PRIM_SPHERE)
            {
                newPrimArray.push_back(primNode);
            }
        }

        // Copy the rest
        if(flags[PACK_GFG])
        {
            // Add the rest as a single file
            nlohmann::json packedPrimNode;
            auto gfgPath = fs::path(outFileName).replace_extension(fs::path(GFG_TAG));
            auto relGFG = fs::relative(gfgPath, fs::path(outFileName).remove_filename());

            packedPrimNode[NodeNames::FILE] = relGFG.generic_string();
            packedPrimNode[NodeNames::TAG] = GFG_TAG;
            // TODO: Change this
            packedPrimNode[NodeNames::TYPE] = "Triangle";

            uint32_t globalInnerIndexCounter = 0;
            bool isSinglePrim = (parsedMeshes.size() == 1 &&
                                 parsedMeshes.front().primIds.size() == 1);
            if(isSinglePrim)
            {
                packedPrimNode[NodeNames::ID] = parsedMeshes.front().primIds.front();
                packedPrimNode[NodeNames::INNER_INDEX] = 0;
            }
            else for(const auto & m : parsedMeshes)
            {
                assert(m.primIds.size() == m.innerIndices.size());
                for(size_t i = 0; i < m.primIds.size(); i++)
                {
                    packedPrimNode[NodeNames::ID].push_back(m.primIds[i]);
                    packedPrimNode[NodeNames::INNER_INDEX].push_back(globalInnerIndexCounter);
                    globalInnerIndexCounter++;
                }
            }
            newPrimArray.push_back(packedPrimNode);
        }
        else for(const auto& m : parsedMeshes)
        {
            std::string meshPath = Filesystem::RelativePathToAbsolute(m.filePath,
                                                                      inScenePath);

            nlohmann::json primNode;
            auto gfgPath = fs::path(meshPath).replace_extension(fs::path(GFG_TAG));
            auto relGFG = fs::relative(gfgPath, fs::path(outFileName).remove_filename());
            primNode[NodeNames::FILE] = relGFG.generic_string();
            primNode[NodeNames::TAG] = GFG_TAG;

            assert(m.primIds.size() == m.innerIndices.size());
            if(m.primIds.size() == 1)
            {
                primNode[NodeNames::ID] = (m.primIds.front());
                primNode[NodeNames::INNER_INDEX] = (m.innerIndices.front());
            }
            else for(size_t i = 0; i < m.primIds.size(); i++)
            {
                primNode[NodeNames::ID].push_back(m.primIds[i]);
                primNode[NodeNames::INNER_INDEX].push_back(m.innerIndices[i]);
            }
            newPrimArray.push_back(primNode);
        }

        // Change the prim list
        outJson[NodeNames::PRIMITIVE_LIST] = newPrimArray;


        // There may be textures, their files should be relative to this
        // file as well
        if(auto tLoc = outJson.find(NodeNames::TEXTURE_LIST);
           tLoc != outJson.end())
        {
            for(auto& tex : *tLoc)
            {
                auto& files = tex.at(NodeNames::FILE);
                if(files.is_string())
                {
                    std::string texPath = files.get<std::string>();
                    std::string texPathAbs = Filesystem::RelativePathToAbsolute(texPath,
                                                                                inScenePath);
                    auto outScenePath = fs::path(outFileName).remove_filename().generic_string();
                    std::string texRelPath = fs::relative(fs::path(texPathAbs),
                                                          outScenePath).generic_string();
                    files = texRelPath;
                }
                else for(auto& f : files)
                {
                    std::string texPath = f.get<std::string>();
                    std::string texPathAbs = Filesystem::RelativePathToAbsolute(texPath,
                                                                                inScenePath);
                    auto outScenePath = fs::path(outFileName).remove_filename().generic_string();
                    std::string texRelPath = fs::relative(fs::path(texPathAbs),
                                                          outScenePath).generic_string();
                    f = texRelPath;
                }
            }
        }
        // Write
        fs::create_directories(fs::absolute(fs::path(outFileName)).remove_filename());
        std::ofstream outFile(outFileName);
        outFile << outJson.dump(4);
        // All Done!
    }
    catch(const MRayError& e)
    {
        return e;
    }
    // Json related errors
    catch(const nlohmann::json::exception& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }

    timer.Split();
    return timer.Elapsed<Second>();
}