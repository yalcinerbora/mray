#include "GFGConverter.h"

#include <filesystem>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <gfg/GFGFileExporter.h>

#include <nlohmann/json.hpp>

#include "Core/Timer.h"
#include "Core/MemAlloc.h"
#include "Core/Vector.h"
#include "Core/Quaternion.h"

// TODO: Type leak change it? (This functionaly is
// highly intrusive, probably needs a redesign?)
#include "SceneLoaderMRay/NodeNames.h"

static const std::string ASSIMP_TAG = "assimp";
static const std::string GFG_TAG = "gfg";

std::string  SceneRelativePathToAbsolute(std::string_view sceneRelativePath,
                                         std::string_view scenePath)
{
    using namespace std::filesystem;
    // Skip if path is absolute
    if(path(sceneRelativePath).is_absolute())
        return std::string(sceneRelativePath);
    // Create an absolute path relative to the scene.json file
    path fullPath = path(scenePath) / path(sceneRelativePath);
    return absolute(fullPath).string();
}

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
static_assert(MemoryC<VectorBackedMemory>, "");

MRAY_GFGCONVERTER_ENTRYPOINT
Expected<double> MRayConvert::ConvertMeshesToGFG(const std::string& outFileName,
                                                 const std::string& inFileName,
                                                 ConversionFlags flags)
{
    using enum ConvFlagEnum;
    namespace fs = std::filesystem;
    if(flags[FAIL_ON_OVERWRITE] &&
       fs::exists(fs::path(outFileName)))
    {
        return MRayError("\"{}\" already exists!",
                         outFileName);
    }

    std::string inScenePath = fs::path(inFileName).remove_filename().string();

    // Do the loading
    Timer timer; timer.Start();
    Expected<nlohmann::json> sceneJsonE = OpenFile(inFileName);
    if(sceneJsonE.has_error())
        return sceneJsonE.error();

    const auto& sceneJson = sceneJsonE.value();

    try
    {
        const auto& primNodes = sceneJson.at(NodeNames::PRIMITIVE_LIST);
        // Arbitrarly reserve some, to prevent many early mallocs
        std::vector<MeshGroup> parsedMeshes;
        parsedMeshes.reserve(512);
        for(const auto& primNode : primNodes)
        {
            const auto& tag = primNodes.at(NodeNames::TAG).get<std::string_view>();
            const auto& idList = primNode.at(NodeNames::ID);
            const auto& innerIndexList = primNode.at(NodeNames::INNER_INDEX);

            // Skip in node triangles, spheres
            if(tag == NodeNames::NODE_PRIM_TRI ||
               tag == NodeNames::NODE_PRIM_TRI_INDEXED ||
               tag == NodeNames::NODE_PRIM_SPHERE)
                continue;

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

        // Acquired all the meshes, and their prim ids
        // Do not immidiately start converting
        // first check if all the files
        //
        // Might aswell get the single name here after calculation
        std::string outGFGNameSingle;
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
                                     outGFGPath.string());
            }
            else for(const auto& mesh : parsedMeshes)
            {
                std::string meshPath = SceneRelativePathToAbsolute(mesh.filePath,
                                                                   inScenePath);
                auto gfgPath = fs::path(meshPath).replace_extension(fs::path(GFG_TAG));

                if(fs::exists(gfgPath))
                    return MRayError("GFG file \"{}\" already exists",
                                     gfgPath.string());
            }
        }

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


        GFGFileExporter exporter;
        for(const auto& mesh : parsedMeshes)
        {
            std::string meshPath = SceneRelativePathToAbsolute(mesh.filePath,
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
                header.aabb.max[0] = meshIn->mAABB.mMax[0];
                header.aabb.max[1] = meshIn->mAABB.mMax[1];
                header.aabb.max[2] = meshIn->mAABB.mMax[2];

                header.componentCount = (flags[NORMAL_AS_QUATERNION])
                                         ? 3 : 5;

                header.indexCount = indexCount;
                header.indexSize = sizeof(uint32_t);
                header.topology = GFGTopology::TRIANGLE;
                header.vertexCount = vertexCount;

                std::vector<GFGVertexComponent> components;
                components.reserve(5);
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
                                         .dataType = GFGDataType::FLOAT_3,
                                         .logic = GFGVertexComponentLogic::UV,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 3
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


                VectorBackedMemory indexMem;
                Span<Vector3ui> indices;
                MemAlloc::AllocateMultiData(std::tie(indices), indexMem,
                                            {indexCount}, 0);

                VectorBackedMemory meshMem;
                Span<Vector3> positions;
                Span<Vector2> uvs;
                Span<Quaternion> normalsQuat;
                Span<Vector3> normals;
                Span<Vector3> tangents;
                Span<Vector3> bitangents;

                if(flags[NORMAL_AS_QUATERNION])
                {
                    MemAlloc::AllocateMultiData(std::tie(positions, uvs,
                                                         normalsQuat),
                                                meshMem,
                                                {vertexCount, vertexCount,
                                                vertexCount},
                                                0u);
                }
                else
                {
                    MemAlloc::AllocateMultiData(std::tie(positions, uvs,
                                                         normals, tangents, bitangents),
                                                meshMem,
                                                {vertexCount, vertexCount,
                                                vertexCount, vertexCount,
                                                vertexCount},
                                                0u);
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
                        normalsQuat[i] = TransformGen::ToSpaceQuat(t, b, n);
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
                exporter.AddMesh(0, components,
                                 header, meshMem.v,
                                 &(indexMem.v));
            }

            if(!flags[PACK_GFG])
            {
                auto gfgPath = fs::path(meshPath).replace_extension(fs::path(GFG_TAG));
                std::ofstream outFile(gfgPath);
                GFGFileWriterSTL fileWriter(outFile);
                exporter.Write(fileWriter);
                exporter.Clear();
            }
        }

        if(flags[PACK_GFG])
        {
            auto gfgPath = fs::path(outFileName).replace_extension(fs::path(GFG_TAG));
            std::ofstream outFile(gfgPath);
            GFGFileWriterSTL fileWriter(outFile);
            exporter.Write(fileWriter);
            exporter.Clear();
        }

        // Write complete
        // Now we need to convert the json file
        // Copy the file representation
        nlohmann::json outJson = sceneJson;
        // Add in node primitives
        nlohmann::json newPrimArray;
        for(const auto& primNode : primNodes)
        {
            const auto& tag = primNodes.at(NodeNames::TAG).get<std::string_view>();
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
            packedPrimNode[NodeNames::FILE] = gfgPath.string();
            packedPrimNode[NodeNames::TAG] = GFG_TAG;
            // TODO: Change this
            packedPrimNode[NodeNames::TYPE] = "Triangle";

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
                    packedPrimNode[NodeNames::INNER_INDEX].push_back(m.innerIndices[i]);
                }
            }
            newPrimArray.push_back(packedPrimNode);
        }
        else for(const auto& m : parsedMeshes)
        {
            std::string meshPath = SceneRelativePathToAbsolute(m.filePath,
                                                               inScenePath);

            nlohmann::json primNode;
            auto gfgPath = fs::path(meshPath).replace_extension(fs::path(GFG_TAG));
            primNode[NodeNames::FILE] = gfgPath.string();
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

        // Write
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
    return timer.Elapsed<Millisecond>();
}