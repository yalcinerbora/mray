#include "MeshProcessor.h"

#include "MRayUSDTypes.h"

#include "Core/Log.h"
#include "Core/TracerI.h"
#include "Core/TypeNameGenerators.h"
#include "Core/ShapeFunctions.h"

#include <barrier>

#include <BS/BS_thread_pool.hpp>

#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/subset.h>

const pxr::TfToken MeshProcessorThread::uvToken = pxr::TfToken("st");
const pxr::TfToken MeshProcessorThread::normalsToken = pxr::TfToken("normals");


// At most 16-gon poly is supported
static constexpr uint32_t MRAY_USD_MAX_TRI_POLY_COUNT = 16;
template<class T>
using TriangulateFaceArray = std::array<T, MRAY_USD_MAX_TRI_POLY_COUNT>;
using TriangulateIndexArray = std::array<Vector3ui, MRAY_USD_MAX_TRI_POLY_COUNT - 2>;
using TriangulatePosArray = TriangulateFaceArray<Vector3>;

bool MeshProcessorThread::IsIndicesPerVertex(const pxr::TfToken& t)
{
    return (t == pxr::UsdGeomTokens->vertex ||
            t == pxr::UsdGeomTokens->varying);
};

enum class IndexMode
{
    PER_VERTEX,
    PER_VERTEX_WITH_INDEX,
    //
    PER_VERTEX_PER_PRIM,
    PER_VERTEX_PER_PRIM_WITH_INDEX,
    //
    CONSTANT
};

class AttributeIndexer
{
    private:
    const pxr::VtArray<int>& posIndices;
    const pxr::VtArray<int>& attributeIndices;
    IndexMode mode;

    public:
    AttributeIndexer(const pxr::VtArray<int>& posIndicesIn,
                     const pxr::VtArray<int>& attributeIndicesIn,
                     const pxr::TfToken& layout);
    uint32_t operator()(uint32_t perVertexPerFaceCounter) const;
};

inline AttributeIndexer::AttributeIndexer(const pxr::VtArray<int>& posIndicesIn,
                                          const pxr::VtArray<int>& attributeIndicesIn,
                                          const pxr::TfToken& layout)
    : posIndices(posIndicesIn)
    , attributeIndices(attributeIndicesIn)
{
    bool hasIndex = !attributeIndices.empty();
    using enum IndexMode;
    if(layout == pxr::UsdGeomTokens->vertex ||
       layout == pxr::UsdGeomTokens->varying)
        mode = hasIndex ? PER_VERTEX_WITH_INDEX : PER_VERTEX;
    else if(layout == pxr::UsdGeomTokens->faceVarying)
        mode = hasIndex ? PER_VERTEX_PER_PRIM_WITH_INDEX
                        : PER_VERTEX_PER_PRIM;
    else // constant
        mode = CONSTANT;
}

inline uint32_t AttributeIndexer::operator()(uint32_t perVertexPerFaceCounter) const
{
    switch(mode)
    {
        using enum IndexMode;
        // Attribute is laid out in per vertex fashion
        // To access we need to check which vertex it points to
        // via position indices
        case PER_VERTEX:
            return uint32_t(posIndices[perVertexPerFaceCounter]);
            // Most complicated one, the **index** of the
            // attribute is laid out per-vertex. Find the vertex
            // then find the index
        case PER_VERTEX_WITH_INDEX:
            return uint32_t(attributeIndices[uint32_t(posIndices[perVertexPerFaceCounter])]);
            // Data is laid out same as the index counter,
            // no need to process "counter is index".
        case PER_VERTEX_PER_PRIM:
            return uint32_t(perVertexPerFaceCounter);
            // Data's **index** is laid out same as the index counter.
            // Do single indirection
        case PER_VERTEX_PER_PRIM_WITH_INDEX:
            return uint32_t(attributeIndices[perVertexPerFaceCounter]);
        case CONSTANT:
            return 0u;
        default:
        {
            assert(false);
            return 0;
        }
    }
}

bool Triangulate(Span<Vector3ui> localIndicesOut,
                 Span<const Vector3> vertices)
{
    using Shape::Polygon::ClipEars;
    // Doing this static functions to enable some unrolling etc.
    // Probably not worth it but w/e
    #define MRAY_CLIP_EAR_GEN(N) case N: \
        ClipEars<N>(Span<Vector3ui, N-2>(localIndicesOut), \
                    Span<const Vector3, N>(vertices)); break

    assert(localIndicesOut.size() + 2 == vertices.size());
    switch(vertices.size())
    {
        case 3:
            localIndicesOut.front() = Vector3ui(0, 1, 2);
            break;
        // Macro generated cases
        MRAY_CLIP_EAR_GEN(4);
        MRAY_CLIP_EAR_GEN(5);
        MRAY_CLIP_EAR_GEN(6);
        MRAY_CLIP_EAR_GEN(7);
        MRAY_CLIP_EAR_GEN(8);
        MRAY_CLIP_EAR_GEN(9);
        MRAY_CLIP_EAR_GEN(10);
        MRAY_CLIP_EAR_GEN(11);
        MRAY_CLIP_EAR_GEN(12);
        MRAY_CLIP_EAR_GEN(13);
        MRAY_CLIP_EAR_GEN(14);
        MRAY_CLIP_EAR_GEN(15);
        MRAY_CLIP_EAR_GEN(16);
        default: return true;
    }
    #undef MRAY_CLIP_EAR_GEN
    return false;
}

Vector3 CalculateTriangleTangent(Vector3ui triIndex, uint32_t vertexIndex,
                                 const TriangulateFaceArray<Vector3>& positions,
                                 const TriangulateFaceArray<Vector3>& normals,
                                 const TriangulateFaceArray<Vector2>& uvs)
{
    switch(vertexIndex)
    {
        case 1: triIndex = Vector3ui(triIndex[0], triIndex[1], triIndex[2]); break;
        case 2: triIndex = Vector3ui(triIndex[0], triIndex[1], triIndex[2]); break;
        default: break;
    }
    Vector3 tangent = Shape::Triangle::CalculateTangent(positions[triIndex[0]],
                                                        positions[triIndex[1]],
                                                        positions[triIndex[2]],
                                                        uvs[triIndex[0]],
                                                        uvs[triIndex[1]],
                                                        uvs[triIndex[2]]);
    return Graphics::GSOrthonormalize(tangent, normals[triIndex[0]]);
}


MRayError MeshProcessorThread::AllocateTransientBuffers(Span<Vector3ui>& indexBuffer, Span<Vector3>& posBuffer,
                                                        Span<Quaternion>& normalBuffer, Span<Vector2>& uvBuffer,
                                                        SubGeomTransientData& transientDataList,
                                                        uint32_t primCount, uint32_t attributeCount)
{
    uint32_t indexAttribI = std::numeric_limits<uint32_t>::max();
    uint32_t posAttribI = std::numeric_limits<uint32_t>::max();
    uint32_t normalAttribI = std::numeric_limits<uint32_t>::max();
    uint32_t uvAttribI = std::numeric_limits<uint32_t>::max();
    // TODO: Now we need to do a type dynamic stuff here,
    // but we dont do it right now so do a runtime check and crash the system for now
    // I am overloaded from the dynamics of USD I did not bother with this.
    uint32_t attribIndex = 0;
    for(const PrimAttributeInfo& attribInfo : mrayPrimAttribInfoList)
    {
        PrimitiveAttributeLogic logic = std::get<PrimAttributeInfo::LOGIC_INDEX>(attribInfo);
        MRayDataTypeRT dataType = std::get<PrimAttributeInfo::LAYOUT_INDEX>(attribInfo);
        MRayError err = std::visit([&](auto&& type) -> MRayError
        {
            using DataType = typename std::remove_cvref_t<decltype(type)>::Type;
            size_t elementCount = (logic == PrimitiveAttributeLogic::INDEX)
                                    ? primCount : attributeCount;
            transientDataList.emplace_back(std::in_place_type_t<DataType>{}, elementCount);
            transientDataList.back().ReserveAll();
            switch(logic)
            {
                using enum PrimitiveAttributeLogic;
                case INDEX:     indexAttribI    = attribIndex; break;
                case POSITION:  posAttribI      = attribIndex; break;
                case UV0:       uvAttribI       = attribIndex; break;
                case NORMAL:    normalAttribI   = attribIndex; break;
                default: break;
            }

            using enum PrimitiveAttributeLogic;
            bool fatalCrash = (logic == INDEX    && !std::is_same_v<Vector3ui, DataType> ||
                               logic == POSITION && !std::is_same_v<Vector3, DataType>   ||
                               logic == UV0      && !std::is_same_v<Vector2, DataType>   ||
                               logic == NORMAL   && !std::is_same_v<Quaternion, DataType>);
            if(fatalCrash)
            {
                return MRayError("[Fatal Error!]: MRayUSD's Triangle Layout is different "
                                 "from Tracer's triangle layout");
            }
            return MRayError::OK;
        }, dataType);
        //
        if(err) return err;
        //
        attribIndex++;
    }
    // Transient buffers are allocated, now fill them
    indexBuffer = transientDataList[indexAttribI].AccessAs<Vector3ui>();
    posBuffer = transientDataList[posAttribI].AccessAs<Vector3>();
    normalBuffer = transientDataList[normalAttribI].AccessAs<Quaternion>();
    uvBuffer = transientDataList[uvAttribI].AccessAs<Vector2>();
    return MRayError::OK;
}

MRayError MeshProcessorThread::TriangulateAndCalculateTangents(uint32_t subgeomIndex, bool changeWinding,
                                                               const AttributeIndexer& posIndexer,
                                                               const AttributeIndexer& uvIndexer,
                                                               const AttributeIndexer& normalIndexer,
                                                               const pxr::VtArray<int>& faceIndices,
                                                               const pxr::VtArray<int>& faceIndexOffsets,
                                                               const pxr::VtArray<pxr::GfVec3f>& positions,
                                                               const pxr::VtArray<pxr::GfVec3f>& normals,
                                                               const pxr::VtArray<pxr::GfVec2f>& uvs)
{
    uint32_t indexCounter = 0;
    uint32_t attribCounter = 0;
    for(int faceIndex : faceIndices.AsConst())
    {
        uint32_t faceIndexStart = (faceIndex == 0) ? 0u : uint32_t(faceIndexOffsets[faceIndex - 1]);
        uint32_t faceIndexEnd = uint32_t(faceIndexOffsets[faceIndex]);
        uint32_t faceVertexCount = uint32_t(faceIndexEnd - faceIndexStart);
        bool failTriangulation = faceVertexCount > MRAY_USD_MAX_TRI_POLY_COUNT;
        // We ignore, if poly is too large
        faceVertexCount = std::min(faceVertexCount, MRAY_USD_MAX_TRI_POLY_COUNT);
        uint32_t faceTriCount = faceVertexCount - 2u;

        // Load stuff to local buffer for processing
        TriangulateIndexArray localIndicesTriangulated;
        TriangulateFaceArray<Vector3> localPositions;
        TriangulateFaceArray<Vector3> localNormals;
        TriangulateFaceArray<Vector2> localUVs;
        //
        TriangulateFaceArray<uint32_t> usdUVIndices;
        TriangulateFaceArray<uint32_t> usdNormalIndices;
        TriangulateFaceArray<uint32_t> usdPosIndices;
        // Load the indices
        for(uint32_t i = 0; i < faceVertexCount; i++)
        {
            uint32_t indexOffset  = faceIndexStart + i;
            usdUVIndices[i]     = uvIndexer(indexOffset);
            usdNormalIndices[i] = normalIndexer(indexOffset);
            usdPosIndices[i]    = posIndexer(indexOffset);
        }
        // Change the winding if requested
        if(changeWinding)
        {
            std::reverse(usdUVIndices.begin(), usdUVIndices.begin() + faceVertexCount);
            std::reverse(usdNormalIndices.begin(), usdNormalIndices.begin() + faceVertexCount);
            std::reverse(usdPosIndices.begin(), usdPosIndices.begin() + faceVertexCount);
        }

        // These are guaranteed to be available.
        for(uint32_t i = 0; i < faceVertexCount; i++)
        {
            pxr::GfVec3f pos = positions[usdPosIndices[i]];
            localPositions[i] = Vector3(pos[0], pos[1], pos[2]);
        }

        // First off triangulate the mesh
        // Skip if we can't triangulate (either 1,2 or more than 16 vertices)
        failTriangulation = Triangulate(Span(localIndicesTriangulated.data(), faceTriCount),
                                        Span(localPositions.data(), faceVertexCount));
        if(failTriangulation)
        {
            warnFailTriangulation = true;
            continue;
        }

        // Normals are not guaranteed
        // So set something reasonable
        if(normals.empty())
        {
            // Set to face normal (assuming vertices are planar)
            Vector3 p0 = localPositions[localIndicesTriangulated[0][0]];
            Vector3 p1 = localPositions[localIndicesTriangulated[0][1]];
            Vector3 p2 = localPositions[localIndicesTriangulated[0][2]];
            Vector3 e0 = p1 - p0;
            Vector3 e1 = p2 - p0;
            Vector3 n = Vector3::Cross(e0, e1).Normalize();
            std::fill(localNormals.begin(),
                      localNormals.begin() + faceVertexCount,
                      n);
        }
        else for(uint32_t i = 0; i < faceVertexCount; i++)
        {
            pxr::GfVec3f n = normals[usdNormalIndices[i]];
            localNormals[i] = Vector3(n[0], n[1], n[2]);
        }
        // We will need uv's for tangent space calcuation
        // Write something reasonable
        for(uint32_t i = 0; i < faceVertexCount; i++)
        {
            if(uvs.empty())
            {
                // In this case, we do simple planar mapping-like approach
                // Rotate each vertex poisition to tangent space and get (x,y)
                // After that, normalize the coordinates (no need for normalization,
                // but it at least be between [0,1]).
                Quaternion rot = Quaternion::RotationBetweenZAxis(localNormals[i]);
                localUVs[i] = Vector2(rot.ApplyRotation(localPositions[i])).Normalize();
            }
            else
            {
                pxr::GfVec2f uv = uvs[usdUVIndices[i]];
                localUVs[i] = Vector2(uv[0], uv[1]);
            }
        }

        // Now calculate tangents while writing
        for(uint32_t i = 0; i < faceTriCount; i++)
        {
            Vector3ui outIndex;
            for(uint32_t j = 0; j < 3; j++)
            {
                uint32_t localIndex = localIndicesTriangulated[i][j];
                IndexTriplet indexTriplet =
                {
                    usdPosIndices[localIndex],
                    usdUVIndices[localIndex],
                    usdNormalIndices[localIndex]
                };
                auto [indexLoc, isInserted] = indexLookupTable.emplace(indexTriplet, indexCounter);
                outIndex[j] = indexLoc->second;

                if(isInserted)
                {
                    indexCounter++;
                    triangleDataTangents.push_back(Vector3::Zero());
                    triangleDataNormals.push_back(localNormals[localIndex]);
                    usdDataIndices.push_back(indexTriplet);
                }
            }
            triangleIndices.push_back(outIndex);

            triangleDataTangents[outIndex[0]] +=
                CalculateTriangleTangent(localIndicesTriangulated[i], 0,
                                         localPositions, localNormals, localUVs);

            triangleDataTangents[outIndex[1]] +=
                CalculateTriangleTangent(localIndicesTriangulated[i], 1,
                                         localPositions, localNormals, localUVs);

            triangleDataTangents[outIndex[2]] +=
                CalculateTriangleTangent(localIndicesTriangulated[i], 2,
                                         localPositions, localNormals, localUVs);
        }
    }
    // Write prim locals
    primLocalPrimCounts[subgeomIndex] = PrimCount
    {
        .primCount = uint32_t(triangleIndices.size()),
        .attributeCount = uint32_t(triangleDataTangents.size())
    };

    Span<Vector3ui> indexBuffer;
    Span<Vector3> posBuffer;
    Span<Quaternion> normalBuffer;
    Span<Vector2> uvBuffer;
    MRayError err = AllocateTransientBuffers(indexBuffer, posBuffer,
                                             normalBuffer, uvBuffer,
                                             primTransientData.back()[subgeomIndex],
                                             primLocalPrimCounts[subgeomIndex].primCount,
                                             primLocalPrimCounts[subgeomIndex].attributeCount);
    if(err) return err;

    // Indirect copy the uv and positions
    for(const auto& indexTriplet : usdDataIndices)
    {
        pxr::GfVec3f pos = positions[indexTriplet[0]];
        pxr::GfVec2f uv = uvs[indexTriplet[1]];
        posBuffer[attribCounter] = Vector3(pos[0], pos[1], pos[2]);
        uvBuffer[attribCounter] = Vector2(uv[0], uv[1]);
    }
    // Memcpy the indices
    assert(triangleIndices.size() == indexBuffer.size());
    std::copy(triangleIndices.cbegin(), triangleIndices.cend(), indexBuffer.begin());

    // Calculate the Quaternion
    assert(triangleDataNormals.size() == triangleDataTangents.size());
    for(size_t i = 0; i < triangleDataNormals.size(); i++)
    {
        Vector3 t = triangleDataTangents[i].Normalize();
        Vector3 n = triangleDataTangents[i];
        Vector3 b = Vector3::Cross(t, n);
        normalBuffer[i] = TransformGen::ToSpaceQuat(t, b, n);
    }

    MRAY_LOG("[{}] After single-index triangulation: primCount {}, vertexCount {}",
             subgeomIndex, primLocalPrimCounts[subgeomIndex].primCount,
             primLocalPrimCounts[subgeomIndex].attributeCount);
    // Finally, after days of work
    // All Done! (Hopefully)
    return MRayError::OK;
}

MRayError MeshProcessorThread::PreprocessIndicesSingle(uint32_t index)
{
    primTransientData.emplace_back();

    using PrimVar = pxr::UsdGeomPrimvar;
    using Attribute = pxr::UsdAttribute;
    // Process the mesh to find the prim counts
    pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh(flatUniques[index]);
    pxr::UsdGeomPrimvarsAPI primVars(flatUniques[index]);

    MRAY_LOG("Processing {}...", flatUniques[index].GetPath().GetString());

    Attribute vertexPosA = mesh.GetPointsAttr();
    Attribute faceVCountA = mesh.GetFaceVertexCountsAttr();
    Attribute faceVIndexA = mesh.GetFaceVertexIndicesAttr();
    PrimVar uvPV = primVars.FindPrimvarWithInheritance(uvToken);
    PrimVar normalPV = primVars.FindPrimvarWithInheritance(normalsToken);

    // Unfortunately usd has a complex indexing scheme,
    // we need to convert this to a single-indexed scheme
    // However; this needs accessing the data (at least the index).
    // Since we already access the indices, might as well store it
    // this can be a memory hog we need to check this later.
    pxr::VtArray<int> vertexIndices; faceVIndexA.Get(&vertexIndices);
    pxr::VtArray<int> uvIndices; uvPV.GetIndices(&uvIndices);
    pxr::VtArray<int> normalIndices; normalPV.GetIndices(&normalIndices);
    AttributeIndexer posIndexer(vertexIndices, vertexIndices, pxr::UsdGeomTokens->faceVarying);
    AttributeIndexer uvIndexer(vertexIndices, uvIndices, uvPV.GetInterpolation());
    AttributeIndexer normalIndexer(vertexIndices, normalIndices, normalPV.GetInterpolation());

    // We need to have face offsets, generate.
    pxr::VtArray<int> faceIndexOffsetsIn;
    faceVCountA.Get(&faceIndexOffsetsIn);
    std::inclusive_scan(faceIndexOffsetsIn.cbegin(), faceIndexOffsetsIn.cend(),
                        faceIndexOffsetsIn.begin(), std::plus<int>());
    const auto& faceIndexOffsets = faceIndexOffsetsIn;

    // And to triangulate, we need the vertex positions.
    pxr::VtArray<pxr::GfVec3f> positions;
    vertexPosA.Get(&positions);
    // Well... we need to calculate tangent space so uv's and normals
    // So get everything.
    pxr::VtArray<pxr::GfVec3f> normals;
    normalPV.GetAttr().Get(&normals);
    pxr::VtArray<pxr::GfVec2f> uvs;
    uvPV.GetAttr().Get(&uvs);

    MRAY_LOG("Face Count: Input {}", faceIndexOffsets.size());
    MRAY_LOG("Pos: IndexCount({}), Count({})", vertexIndices.size(), positions.size());
    MRAY_LOG("UV: IndexCount({}), Count({}), Interp({})",
             uvIndices.size(), uvs.size(), uvPV.GetInterpolation().GetString());
    MRAY_LOG("Normal: IndexCount({}), Count({}), Interp({})",
             normalIndices.size(), normals.size(), normalPV.GetInterpolation().GetString());
    MRAY_LOG("---------------------------");

    // Orientation
    pxr::TfToken orientation;
    mesh.GetOrientationAttr().Get(&orientation);
    bool changeToCW = (orientation == pxr::UsdGeomTokens->leftHanded);

    // Reset the buffers
    auto subsets = pxr::UsdGeomSubset::GetAllGeomSubsets(mesh);
    primLocalPrimCounts.resize(std::max(size_t(1), subsets.size()));
    primTransientData.back().resize(std::max(size_t(1), subsets.size()));
    if(subsets.empty())
    {
        // Reset the hash table & buffers
        indexLookupTable.clear();
        usdDataIndices.clear();
        triangleDataTangents.clear();
        triangleDataNormals.clear();
        triangleIndices.clear();
        // To reuse the complex triangulation + single-indexer function
        // we create iota here.
        // TODO: Abstract this to a ranges view later
        // (a python generator-like concept)
        pxr::VtArray<int> faceIndices(faceIndexOffsetsIn.size());
        std::iota(faceIndices.begin(), faceIndices.end(), 0);

        MRAY_LOG("    No Subset: FaceCount {}", faceIndices.size());

        // All the work is here
        MRayError err = TriangulateAndCalculateTangents(0, changeToCW,
                                                        posIndexer, uvIndexer,
                                                        normalIndexer, faceIndices,
                                                        faceIndexOffsets, positions,
                                                        normals, uvs);
        if(err) return err;
    }
    else for(uint32_t i = 0; i < subsets.size(); i++)
    {
        const auto& subset = subsets[i];
        // Reset the hash table & buffers
        indexLookupTable.clear();
        usdDataIndices.clear();
        triangleDataTangents.clear();
        triangleDataNormals.clear();
        triangleIndices.clear();
        // Get the face indices
        pxr::VtArray<int> faceIndices;
        subset.GetIndicesAttr().Get(&faceIndices);

        MRAY_LOG("    {: >2} Subset: FaceCount {}", i, faceIndices.size());

        // All the work is here
        MRayError err = TriangulateAndCalculateTangents(i, changeToCW ,
                                                        posIndexer, uvIndexer,
                                                        normalIndexer, faceIndices,
                                                        faceIndexOffsets, positions,
                                                        normals, uvs);
        if(err) return err;
    }
    // Fully reserve the batches
    primBatchOutputs[index] = tracer.ReservePrimitiveBatches(primGroupId, primLocalPrimCounts);

    return MRayError::OK;
}

MRayError MeshProcessorThread::LoadMeshDataSingle(uint32_t index)
{
    std::vector<SubGeomTransientData>& primGeomData = primTransientData[index];
    const std::vector<PrimBatchId>& primBatchIds = primBatchOutputs[index];
    assert(primBatchIds.size() == primGeomData.size());

    for(uint32_t i = 0; i < primBatchIds.size(); i++)
    {
        SubGeomTransientData& subGeomData = primGeomData[i];
        PrimBatchId primBatchId = primBatchIds[i];
        for(uint32_t aI = 0; aI < subGeomData.size(); aI++)
        {
            tracer.PushPrimAttribute(primGroupId, primBatchId, aI,
                                     std::move(subGeomData[aI]));
        }
    }
    return MRayError::OK;
}

MRayError MeshProcessorThread::PreprocessIndices()
{
    // Reserve for 2^20 (~1 million) elements beforehand.
    indexLookupTable.reserve(1_MiB);
    usdDataIndices.reserve(1_MiB);
    triangleDataTangents.reserve(1_MiB);
    triangleDataNormals.reserve(1_MiB);
    triangleIndices.reserve(1_MiB);

    MRayError err = MRayError::OK;
    assert(flatUniques.size() == primBatchOutputs.size());
    for(uint32_t i = 0; i < flatUniques.size(); i++)
    {
        // Check the error here, to make the other function
        // more readable
        pxr::TfToken subdivMode;
        pxr::UsdGeomMesh(flatUniques[i]).GetSubdivisionSchemeAttr().Get(&subdivMode);
        if(subdivMode != pxr::UsdGeomTokens->none)
            warnSubdivisionSurface = true;
        // Actual Work
        err = PreprocessIndicesSingle(i);
        if(err) break;
    }
    return err;
}

MRayError MeshProcessorThread::LoadMeshData()
{
    MRayError err = MRayError::OK;
    assert(flatUniques.size() == primBatchOutputs.size());
    for(uint32_t i = 0; i < flatUniques.size(); i++)
    {
        // Actual Work
        err = LoadMeshDataSingle(i);
        if(err) break;
    }
    primTransientData.clear();
    return err;
}

MRayError ProcessUniqueMeshes(// Output
                              std::map<pxr::UsdPrim, std::vector<PrimBatchId>>& outPrimBatches,
                              // I-O
                              TracerI& tracer,
                              BS::thread_pool& threadPool,
                              // Input
                              const CollapsedPrims& meshMatPrims)
{
    uint32_t uniquePrimCount = uint32_t(meshMatPrims.uniquePrims.size());
    std::vector<std::vector<PrimBatchId>> outPrimBatchesFlat;
    outPrimBatchesFlat.resize(uniquePrimCount);

    std::atomic_bool warnSubdivisionSurface = false;
    std::atomic_bool warnTriangulation = false;
    std::atomic_bool warnFailTriangulation = false;

    // Flatten the primitive set
    std::vector<pxr::UsdPrim> flatUniques;
    flatUniques.reserve(uniquePrimCount);
    for(const auto& uniquePrim : meshMatPrims.uniquePrims)
        flatUniques.push_back(uniquePrim);

    // TODO: Statically writing the type name here
    std::string name = TypeNameGen::Runtime::AddPrimitivePrefix("Triangle");
    PrimGroupId primGroupId = tracer.CreatePrimitiveGroup(name);
    PrimAttributeInfoList mrayPrimAttribInfoList = tracer.AttributeInfo(primGroupId);

    const auto BarrierFunc = [&]() noexcept
    {
        try { tracer.CommitPrimReservations(primGroupId); }
        catch(const MRayError& e)
        {
            // Fatally crash here, barrier's sync
            // do not allowed to have exceptions
            MRAY_ERROR_LOG("[Tracer]: {}", e.GetError());
            std::exit(1);
        }
    };
    //uint32_t threadCount = std::min(threadPool.get_thread_count(),
    //                                uniquePrimCount);
    uint32_t threadCount = 1;

    using Barrier = std::barrier<decltype(BarrierFunc)>;
    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

    const auto THRD_ProcessMeshes = [&](uint32_t start, uint32_t end) -> MRayError
    {
        // Subset the data to per core
        std::span myPrimRange(flatUniques.begin() + start, end - start);
        std::span myPrimBatchOutput(outPrimBatchesFlat.begin() + start, end - start);
        MeshProcessorThread meshProcessor
        {
            .tracer = tracer,
            .mrayPrimAttribInfoList = mrayPrimAttribInfoList,
            .flatUniques = myPrimRange,
            .primBatchOutputs = myPrimBatchOutput,
            .primGroupId = primGroupId
        };
        MRayError err = MRayError::OK;
        err = meshProcessor.PreprocessIndices();

        if(meshProcessor.warnSubdivisionSurface)
            warnSubdivisionSurface = true;
        if(meshProcessor.warnTriangulation)
            warnTriangulation = true;
        if(meshProcessor.warnFailTriangulation)
            warnFailTriangulation = true;

        // Sync point, let the barrier function to commit the
        // GPU buffers
        if(err)
        {
            barrier->arrive_and_drop();
            return err;
        }
        barrier->arrive_and_wait();

        // Finally load the mesh data
        err = meshProcessor.LoadMeshData();
        return err;
    };
    auto future = threadPool.submit_blocks(uint32_t(0), uniquePrimCount,
                                           THRD_ProcessMeshes, threadCount);
    future.wait();

    if(warnSubdivisionSurface)
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes have subdivision enabled. MRay only supports "
                         "\"none\" SubdivisionScheme. These are skipped.");
    if(warnTriangulation)
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes have more than 4 face per geometry MRay only supports "
                         "triangle meshes, these are triangulated (Basic ear clipping algorithm, "
                         "do not expect miracles).");

    if(warnFailTriangulation)
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes have either degenerate (1 or 2) vertex or more than 16 vertex "
                         "face. MRay is not a modelling software, fix your mesh.");

    // Concat Errors
    auto errorArray = future.get();
    MRayError err = MRayError::OK;
    bool isFirst = true;
    for(const auto& threadErr : errorArray)
    {
        if(threadErr && isFirst)
            err = threadErr;
        else if(threadErr)
            err.AppendInfo(threadErr.GetError());
    }
    if(err) return err;

    // Copy the batchIds, to map
    assert(flatUniques.size() == outPrimBatchesFlat.size());
    for(size_t i = 0; i < flatUniques.size(); i++)
    {
        outPrimBatches.emplace(flatUniques[i], std::move(outPrimBatchesFlat[i]));
    }
    assert(outPrimBatchesFlat.size() == outPrimBatches.size());
    // All done!
    return err;
}

MRayError  ProcessUniqueSpheres(// Output
                                std::map<pxr::UsdPrim, std::vector<PrimBatchId>>& outPrimBatches,
                                // I-O
                                TracerI& tracer,
                                BS::thread_pool& threadPool,
                                // Input
                                const CollapsedPrims& meshMatPrims)
{
    return MRayError::OK;
}