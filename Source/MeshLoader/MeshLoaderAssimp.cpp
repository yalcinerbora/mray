#include "MeshLoaderAssimp.h"

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>

#include "TransientPool/TransientPool.h"

MeshFileAssimp::MeshFileAssimp(Assimp::Importer& imp,
                               const std::string& fPath,
                               uint32_t internalIndex)
    : importer(imp)
    , filePath(fPath)
    , scene(nullptr)
    , innerIndex(internalIndex)
{
    unsigned int flags = static_cast<unsigned int>
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
    const aiScene* sceneRaw = importer.ReadFile(filePath, flags);
    if(!sceneRaw) throw MRayError("Assimp: Unable to read file \"{}\"", Name());
    scene.reset(importer.GetOrphanedScene());

    if(innerIndex >= scene->mNumMeshes)
        throw MRayError("Inner index is out of range");
}

AABB3 MeshFileAssimp::AABB() const
{
    const auto& aabb = scene->mMeshes[innerIndex]->mAABB;
    return AABB3(Vector3(aabb.mMin.x, aabb.mMin.y, aabb.mMin.z),
                 Vector3(aabb.mMax.x, aabb.mMax.y, aabb.mMax.z));
}

uint32_t MeshFileAssimp::MeshPrimitiveCount() const
{
    const auto& mesh = scene->mMeshes[innerIndex];
    return mesh->mNumFaces;
}

uint32_t MeshFileAssimp::MeshAttributeCount() const
{
    const auto& mesh = scene->mMeshes[innerIndex];
    return mesh->mNumVertices;
}

std::string MeshFileAssimp::Name() const
{
    return std::filesystem::path(filePath).filename().string();
}

TransientData MeshFileAssimp::GetAttribute(PrimitiveAttributeLogic attribLogic) const
{
    static_assert(std::is_same_v<Float, float>,
                  "Currently \"MeshLoaderAssimp\" do not support double "
                  "precision mode change this later.");

    auto PushVertexAttribute = [&]<class T>() -> TransientData
    {
        const auto& mesh = scene->mMeshes[innerIndex];
        size_t localCount = mesh->mNumVertices;
        TransientData input(std::in_place_type_t<T>{}, localCount);

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
                return std::move(input);
            }
            default: return TransientData(std::in_place_type_t<Byte>{}, 0);
        }

        if constexpr(std::is_same_v<Float, double>)
        {

        }

        input.Push(Span<const T>(attributePtr, localCount));
        return std::move(input);
    };

    const auto& mesh = scene->mMeshes[innerIndex];
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
        return std::move(input);
    }
    else
    {
        using enum PrimitiveAttributeLogic;
        switch(attribLogic)
        {
            case POSITION:  return PushVertexAttribute.operator()<Vector3>();
            case NORMAL:    return PushVertexAttribute.operator()<Vector3>();
            case TANGENT:   return PushVertexAttribute.operator()<Vector3>();
            case BITANGENT: return PushVertexAttribute.operator()<Vector3>();
            case UV0:       return PushVertexAttribute.operator()<Vector2>();
            case UV1:       return PushVertexAttribute.operator()<Vector2>();
            default: throw MRayError("Uknown attribute logic");
        }
    }
}

bool MeshFileAssimp::HasAttribute(PrimitiveAttributeLogic attribLogic) const
{
    const auto& mesh = scene->mMeshes[innerIndex];

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

MRayDataTypeRT MeshFileAssimp::AttributeLayout(PrimitiveAttributeLogic attribLogic) const
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

std::unique_ptr<MeshFileI> MeshLoaderAssimp::OpenFile(std::string& filePath,
                                                      uint32_t innerIndex)
{
    return std::unique_ptr<MeshFileI>(new MeshFileAssimp(importer, filePath, innerIndex));
}