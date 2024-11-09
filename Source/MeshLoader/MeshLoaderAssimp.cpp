#include "MeshLoaderAssimp.h"

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>

#include "TransientPool/TransientPool.h"
#include "Core/Error.hpp"

MeshViewAssimp::MeshViewAssimp(uint32_t innerIndexIn, const MeshFileAssimp& fileIn)
    : innerIndex(innerIndexIn)
    , assimpFile(fileIn)
{
    if(innerIndex >= assimpFile.scene->mNumMeshes)
        throw MRayError("Assimp: Inner index is out of range in file \"{}\"",
                        assimpFile.Name());
}

AABB3 MeshViewAssimp::AABB() const
{
    const auto& aabb = assimpFile.scene->mMeshes[innerIndex]->mAABB;
    return AABB3(Vector3(aabb.mMin.x, aabb.mMin.y, aabb.mMin.z),
                 Vector3(aabb.mMax.x, aabb.mMax.y, aabb.mMax.z));
}

uint32_t MeshViewAssimp::MeshPrimitiveCount() const
{
    const auto& mesh = assimpFile.scene->mMeshes[innerIndex];
    return mesh->mNumFaces;
}

uint32_t MeshViewAssimp::MeshAttributeCount() const
{
    const auto& mesh = assimpFile.scene->mMeshes[innerIndex];
    return mesh->mNumVertices;
}

std::string MeshViewAssimp::Name() const
{
    return assimpFile.Name();
}

uint32_t MeshViewAssimp::InnerIndex() const
{
    return innerIndex;
}

// TODO: Report bug to clang. This function was a lambda in
// "MeshViewAssimp::GetAttribute" but it did crash.
template<class T>
TransientData PushVertexAttribute(PrimitiveAttributeLogic attribLogic,
                                  const aiMesh* mesh)
{
    size_t localCount = mesh->mNumVertices;
    TransientData input(std::in_place_type_t<T>{}, localCount);

    // Sanity check (check if aiVector3D has no padding)
    static_assert(sizeof(std::array<aiVector3D, 2>) ==
                    sizeof(Float) * 3 * 2);

    const T* attributePtr;
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:
            attributePtr = reinterpret_cast<const T*>(mesh->mVertices);
            break;
        case NORMAL:
            attributePtr = reinterpret_cast<const T*>(mesh->mNormals);
            break;
        case TANGENT:
            attributePtr = reinterpret_cast<const T*>(mesh->mTangents);
            break;
        case BITANGENT:
            attributePtr = reinterpret_cast<const T*>(mesh->mBitangents);
            break;
        case UV0: case UV1:
        {
            // Manual push for uv
            uint32_t texCoordIndex = (attribLogic == UV0) ? 0 : 1;
            for(unsigned int i = 0; i < mesh->mNumVertices; i++)
            {
                T uv = T(mesh->mTextureCoords[texCoordIndex][i].x,
                            mesh->mTextureCoords[texCoordIndex][i].y);
                input.Push(Span<const T>(&uv, 1));
            }
            return input;
        }
        default: return TransientData(std::in_place_type_t<Byte>{}, 0);
    }
    input.template Push<T>(Span<const T>(attributePtr, localCount));
    assert(input.IsFull());
    return input;
};

TransientData MeshViewAssimp::GetAttribute(PrimitiveAttributeLogic attribLogic) const
{
    static_assert(std::is_same_v<Float, float>,
                  "Currently \"MeshLoaderAssimp\" do not support double "
                  "precision mode change this later.");


    const auto& mesh = assimpFile.scene->mMeshes[innerIndex];
    if(attribLogic == PrimitiveAttributeLogic::INDEX)
    {
        TransientData input(std::in_place_type_t<Vector3ui>{}, mesh->mNumFaces);
        for(unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            const auto& face = mesh->mFaces[i];
            Vector3ui faceIndices(face.mIndices[0],
                                  face.mIndices[1],
                                  face.mIndices[2]);
            input.Push(Span<const Vector3ui>(&faceIndices, 1));
        };
        assert(input.IsFull());
        return input;
    }
    else
    {
        using enum PrimitiveAttributeLogic;
        switch(attribLogic)
        {
            case POSITION:  return PushVertexAttribute<Vector3>(attribLogic, mesh);
            case NORMAL:    return PushVertexAttribute<Vector3>(attribLogic, mesh);
            case TANGENT:   return PushVertexAttribute<Vector3>(attribLogic, mesh);
            case BITANGENT: return PushVertexAttribute<Vector3>(attribLogic, mesh);
            case UV0:       return PushVertexAttribute<Vector2>(attribLogic, mesh);
            case UV1:       return PushVertexAttribute<Vector2>(attribLogic, mesh);
            default: throw MRayError("Uknown attribute logic");
        }
    }
}

bool MeshViewAssimp::HasAttribute(PrimitiveAttributeLogic attribLogic) const
{
    const auto& mesh = assimpFile.scene->mMeshes[innerIndex];

    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case INDEX:     return true;
        case POSITION:  return mesh->HasPositions();
        case NORMAL:    return mesh->HasNormals();
        case BITANGENT:
        case TANGENT:   return mesh->HasTangentsAndBitangents();
        case UV0:       return mesh->HasTextureCoords(0);
        case UV1:       return mesh->HasTextureCoords(1);
        default: return false;
    }
}

MRayDataTypeRT MeshViewAssimp::AttributeLayout(PrimitiveAttributeLogic attribLogic) const
{
    // Assimp always loads to Vector3/2's so no need to check per-inner item etc.
    using enum MRayDataEnum;
    if(attribLogic == PrimitiveAttributeLogic::POSITION ||
       attribLogic == PrimitiveAttributeLogic::NORMAL ||
       attribLogic == PrimitiveAttributeLogic::TANGENT ||
       attribLogic == PrimitiveAttributeLogic::BITANGENT)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_3>());
    else if(attribLogic == PrimitiveAttributeLogic::UV0)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_2>());
    else if(attribLogic == PrimitiveAttributeLogic::UV1)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_2>());
    else if(attribLogic == PrimitiveAttributeLogic::INDEX)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UI>());
    else
        throw MRayError("Unknown attribute logic!");
}

MeshFileAssimp::MeshFileAssimp(Assimp::Importer& imp,
                               const std::string& fPath)
    : filePath(fPath)
    , importer(imp)
    , scene(nullptr)
{
    unsigned int flags = //static_cast<unsigned int>
    //(
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
        // (this guarantees a "mesh" has same-type triangles)
        aiProcess_SortByPType |
        //
        aiProcess_RemoveRedundantMaterials
        ;
    //);
    const aiScene* sceneRaw = importer.ReadFile(filePath, flags);
    if(!sceneRaw) throw MRayError("Assimp: Unable to read file \"{}\"", Name());
    scene.reset(importer.GetOrphanedScene());
}

std::unique_ptr<MeshFileViewI>
MeshFileAssimp::ViewMesh(uint32_t innerIndex)
{
    return std::unique_ptr<MeshFileViewI>(new MeshViewAssimp(innerIndex, *this));
}

std::string MeshFileAssimp::Name() const
{
    return std::filesystem::path(filePath).filename().string();
}

MeshLoaderAssimp::MeshLoaderAssimp()
{
    // TODO: Change this later to utilize skeletal meshes
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE,
                                aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    // Import only position, normal, tangent (if provided) and uv
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                aiComponent_BONEWEIGHTS |
                                aiComponent_COLORS);
}

std::unique_ptr<MeshFileI> MeshLoaderAssimp::OpenFile(std::string& filePath)
{
    return std::unique_ptr<MeshFileI>(new MeshFileAssimp(importer, filePath));
}