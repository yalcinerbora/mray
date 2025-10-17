#include "GFGConverter.h"

#include <filesystem>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/DefaultLogger.hpp>

#include <gfg/GFGFileExporter.h>

#include <nlohmann/json.hpp>

#include "Core/Log.h"
#include "Core/Timer.h"
#include "Core/MemAlloc.h"
#include "Core/Vector.h"
#include "Core/Quaternion.h"
#include "Core/Filesystem.h"
#include "Core/GraphicsFunctions.h"
#include "Core/ThreadPool.h"

// TODO: Type leak change it? (This functionality is
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

struct VectorBackedMemory
{
    std::vector<uint8_t>    v;
    //
    void                    ResizeBuffer(size_t s);
    size_t                  Size() const;
    explicit operator const Byte* () const;
    explicit operator Byte* ();
};

inline void VectorBackedMemory::ResizeBuffer(size_t s)
{
    v.resize(s);
}

inline size_t VectorBackedMemory::Size() const
{
    return v.size();
}

inline VectorBackedMemory::operator const Byte* () const
{
    return reinterpret_cast<const Byte*>(v.data());
}

inline VectorBackedMemory::operator Byte* ()
{
    return reinterpret_cast<Byte*>(v.data());
}

static_assert(MemoryC<VectorBackedMemory>,
              "\"VectorBackedMemory\" does not "
              "satisfy MemoryC concept!");

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

Expected<std::vector<MeshGroup>> FindMeshes(const nlohmann::json& sceneJson)
{
    if(auto primNodes = sceneJson.find(NodeNames::PRIMITIVE_LIST);
       primNodes != sceneJson.end())
    {
        // Arbitrarily reserve some, to prevent many early mallocs
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

void THRDProcessMesh(ErrorList& errors, Span<MeshGroup> meshes,
                     // Used when prims are packed to single mesh
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
    // TODO: GCC warns redundant cast, but MSVC says default enum type is
    // int. so we obey MSVC's warning.
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

    // Thread local buffers
    VectorBackedMemory indexMem;
    VectorBackedMemory meshMem;

    for(auto& mesh : meshes)
    {
        std::string meshPath = Filesystem::RelativePathToAbsolute(mesh.filePath,
                                                                  inScenePath);

        // TODO: Delete previously created files??
        const aiScene* assimpScene = importer.ReadFile(meshPath, assimpFlags);
        if(!assimpScene)
        {
            errors.AddException(MRayError("Assimp: Unable to read file \"{}\"",
                                          meshPath));
            return;
        }

        // GFG require user to lay the data
        // We will push the data as SoA style
        for(unsigned int& meshIndex : mesh.innerIndices)
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

            size_t alignment = Math::Max(alignof(Vector3), alignof(Vector2));
            if(flags[NORMAL_AS_QUATERNION])
                alignment = Math::Max(alignment, alignof(Quaternion));

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
            offset += Math::NextMultiple(vertexCount * components.back().stride, alignment);
            components.push_back(GFGVertexComponent
                                 {
                                     .dataType = GFGDataType::FLOAT_2,
                                     .logic = GFGVertexComponentLogic::UV,
                                     .startOffset = offset,
                                     .internalOffset = 0,
                                     .stride = sizeof(float) * 2
                                 });
            offset += Math::NextMultiple(vertexCount * components.back().stride, alignment);
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
                offset += Math::NextMultiple(vertexCount * components.back().stride, alignment);
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
                offset += Math::NextMultiple(vertexCount * components.back().stride, alignment);
                components.push_back(GFGVertexComponent
                                     {
                                         .dataType = GFGDataType::FLOAT_3,
                                         .logic = GFGVertexComponentLogic::TANGENT,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 3

                                     });
                offset += Math::NextMultiple(vertexCount * components.back().stride, alignment);
                components.push_back(GFGVertexComponent
                                     {
                                         .dataType = GFGDataType::FLOAT_3,
                                         .logic = GFGVertexComponentLogic::BINORMAL,
                                         .startOffset = offset,
                                         .internalOffset = 0,
                                         .stride = sizeof(float) * 3

                                     });
                offset += Math::NextMultiple(vertexCount * components.back().stride, alignment);
            }
            Span<Vector3ui> indices;
            MemAlloc::AllocateMultiData(Tie(indices), indexMem,
                                        {meshIn->mNumFaces}, alignof(Vector3ui));


            Span<Vector3> positions;
            Span<Vector2> uvs;
            Span<Quaternion> normalsQuat;
            Span<Vector3> normals;
            Span<Vector3> tangents;
            Span<Vector3> bitangents;
            if(flags[NORMAL_AS_QUATERNION])
            {
                MemAlloc::AllocateMultiData(Tie(positions, uvs,
                                                normalsQuat),
                                            meshMem,
                                            {vertexCount, vertexCount,
                                             vertexCount},
                                            alignment);
            }
            else
            {
                MemAlloc::AllocateMultiData(Tie(positions, uvs,
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
            static_assert(alignof(Vector3) == alignof(aiVector3D));
            std::memcpy(static_cast<void*>(positions.data()),
                        static_cast<const void*>(meshIn->mVertices),
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
                bool warnDegenerateTangents = false;
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

                    // If the tangents are left-handed,
                    // convert them to right-handed
                    if(Math::Dot(Math::Cross(b, n), t) < Float(0))
                        t = -t;
                    auto [newT, newB] = Graphics::GSOrthonormalize(t, b, n);
                    Quaternion q;
                    if(!(Math::IsFinite(newT) && Math::IsFinite(newB)))
                    {
                        warnDegenerateTangents = true;
                        // If we fail randomly generate space
                        b = Graphics::OrthogonalVector(n);
                        t = Math::Cross(b, n);
                        q = TransformGen::ToSpaceQuat(t, b, n);
                    }
                    else q = TransformGen::ToSpaceQuat(newT, newB, n);
                    normalsQuat[i] = q;
                }

                if(warnDegenerateTangents)
                {
                    MRAY_WARNING_LOG("Mesh File{:s}:[{:d}] has degenerate tangents. "
                                     "These are randomly generated.",
                                     mesh.filePath, meshIndex);
                }
            }
            else
            {
                std::memcpy(static_cast<void*>(normals.data()),
                            static_cast<const void*>(meshIn->mNormals),
                            vertexCount * sizeof(Vector3));
                std::memcpy(static_cast<void*>(tangents.data()),
                            static_cast<const void*>(meshIn->mTangents),
                            vertexCount * sizeof(Vector3));
                std::memcpy(static_cast<void*>(bitangents.data()),
                            static_cast<const void*>(meshIn->mBitangents),
                            vertexCount * sizeof(Vector3));
            }

            // Laid out the data now add to the mesh
            // If global gfg is used, lock it.
            // If is local just add
            if(flags[PACK_GFG])
            {
                // If packed
                std::scoped_lock<std::mutex> lock(exporterMutex);
                meshIndex = exporter.AddMesh(0, components,
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
                                                 uint32_t threadCount,
                                                 ConversionFlags flags)
{
    using TPInitParams = typename ThreadPool::InitParams;
    ThreadPool threadPool(TPInitParams{.threadCount = threadCount});

    using enum ConvFlagEnum;
    namespace fs = std::filesystem;
    if(flags[FAIL_ON_OVERWRITE] &&
       fs::exists(fs::path(outFileName)))
    {
        return MRayError("\"{}\" already exists!",
                         outFileName);
    }

    std::string inScenePath = fs::path(inFileName).remove_filename().generic_string();
    auto outputScenePathAbsolute = fs::absolute(fs::path(outFileName)).remove_filename();

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
        auto& parsedMeshes = meshOut.value();

        // Do file availability validation
        // If don't override flag is set this will give an error
        auto e = ValidateOutputFiles(flags, inFileName, outFileName,
                                     parsedMeshes);
        if(e) return e;


        // Multi-threaded process meshes
        // Normally this wont give much perf, but we do postprocess
        // meshes via assimp to create tangents etc. So this has
        // some gain.
        ErrorList errors;
        Span<MeshGroup> meshes(parsedMeshes);
        std::mutex exporterMutex;
        GFGFileExporter globalExporter;
        auto future = threadPool.SubmitBlocks(uint32_t(parsedMeshes.size()),
                                              [&, meshes, flags](size_t start,
                                                                 size_t end)
        {
            auto mySpan = meshes.subspan(start, end - start);
            GFGFileExporter localExporter;
            GFGFileExporter* expRef = (flags[PACK_GFG]) ? &globalExporter
                                                        : &localExporter;
            THRDProcessMesh(errors, mySpan, *expRef, exporterMutex,
                            inScenePath, flags);
        });

        // Process the errors if any
        future.WaitAll();
        MRayError errOut = MRayError::OK;
        bool hasErrors = false;
        for(MRayError& err : errors.exceptions)
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
            auto relGFG = fs::relative(gfgPath, outputScenePathAbsolute);

            packedPrimNode[NodeNames::FILE] = relGFG.generic_string();
            packedPrimNode[NodeNames::TAG] = GFG_TAG;
            // TODO: Change this, what if the prim type was something else?
            // (User defined custom primitive etc.).
            // We need to store the types and make multiple nodes
            // etc. I did not bothered with it.
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


        // TODO: Currently only primitives and textures have file-backed data.
        // We already processed the primitives and only textures left.
        // In future, we may have other types that will require path conversion.
        //
        // We can currently check if "NodeNames::FILE" exists for all types
        // but this is not a solid process (not every type will have "file", or
        // some other verbose identifier may be used to define a file)
        //
        // There may be textures, their files should be relative to this
        // file as well
        if(auto tLoc = outJson.find(NodeNames::TEXTURE_LIST);
           tLoc != outJson.end())
        {
            for(auto& tex : *tLoc)
            {
                auto& files = tex.at(NodeNames::FILE);
                auto ConvPath = [&](std::string_view texPathRelative)
                {
                    std::string texPathAbs = Filesystem::RelativePathToAbsolute(texPathRelative,
                                                                                inScenePath);
                    std::string newRelPath = fs::relative(fs::path(texPathAbs),
                                                          outputScenePathAbsolute).generic_string();
                    return newRelPath;
                };
                //
                if(files.is_string())
                {
                    files = ConvPath(files.get<std::string_view>());
                }
                else for(auto& f : files)
                {
                    f = ConvPath(f.get<std::string_view>());
                }
            }
        }
        // Write
        fs::create_directories(outputScenePathAbsolute);
        std::ofstream outFile(outFileName);
        // TODO: Cooler output parsing maybe?
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