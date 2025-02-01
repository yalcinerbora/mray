#pragma once

#include <array>
#include <cstdint>

#include "Core/Vector.h"
#include "Core/DataStructures.h"
#include "Core/TracerI.h"

#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/prim.h>

class AttributeIndexer;
struct CollapsedPrims;
namespace BS { class thread_pool;}

using IndexTriplet = std::array<uint32_t, 3>;

struct IndexHasher
{
    size_t operator()(const IndexTriplet& t) const
    {
        std::hash<uint32_t> h;
        return h(t[0]) ^ h(t[1]) ^ h(t[2]);
    }
};

using IndexHashTable = std::unordered_map<IndexTriplet, uint32_t, IndexHasher>;

struct MeshProcessorThread
{
    template<class T>
    using StdVector2D = std::vector<std::vector<T>>;

    using SubGeomTransientData = StaticVector<TransientData, TracerConstants::MaxAttributePerGroup>;

    static const pxr::TfToken uvToken;
    static const pxr::TfToken normalsToken;

    private:
    bool IsIndicesPerVertex(const pxr::TfToken& t);

    MRayError PreprocessIndicesSingle(uint32_t index);
    MRayError LoadMeshDataSingle(uint32_t index);

    MRayError AllocateTransientBuffers(Span<Vector3ui>& indexBuffer, Span<Vector3>& posBuffer,
                                       Span<Quaternion>& normalBuffer, Span<Vector2>& uvBuffer,
                                       SubGeomTransientData& transientData,
                                       uint32_t primCount, uint32_t attributeCount);
    MRayError TriangulateAndCalculateTangents(uint32_t subgeomIndex,
                                              bool changeWinding,
                                              const AttributeIndexer& posIndexer,
                                              const AttributeIndexer& uvIndexer,
                                              const AttributeIndexer& normalIndexer,
                                              const pxr::VtArray<int>& faceIndices,
                                              const pxr::VtArray<int>& faceIndexOffsets,
                                              const pxr::VtArray<pxr::GfVec3f>& positions,
                                              const pxr::VtArray<pxr::GfVec3f>& normals,
                                              const pxr::VtArray<pxr::GfVec2f>& uvs);

    public:
    TracerI&                        tracer;
    const PrimAttributeInfoList&    mrayPrimAttribInfoList;
    Span<const pxr::UsdPrim>        flatUniques;
    Span<std::vector<PrimBatchId>>  primBatchOutputs;
    PrimGroupId                     primGroupId;

    // Intermediates
    // Per-prim
    std::vector<PrimCount>      primLocalPrimCounts;
    // Per-subgeometry
    IndexHashTable              indexLookupTable;
    // Per-subgeometry per triangle vertex
    std::vector<IndexTriplet>   usdDataIndices;
    std::vector<Vector3>        triangleDataTangents;
    std::vector<Vector3>        triangleDataNormals;
    // Per-subgeometry per triangle index
    std::vector<Vector3ui>      triangleIndices;
    // For entire batch
    // TODO: This is too much memory,
    // but we triangulate and resolve multi-index data at the same time.
    // this may be optimized but we touch most of the data.
    // Especially for USD scenes this may be too much.
    // Check this later
    StdVector2D<SubGeomTransientData>   primTransientData;


    //
    bool warnSubdivisionSurface = false;
    bool warnTriangulation      = false;
    bool warnFailTriangulation  = false;

    public:
    MRayError PreprocessIndices();
    MRayError LoadMeshData();
};

// Return something....
MRayError ProcessUniqueMeshes(// Output
                              std::map<pxr::UsdPrim, std::vector<PrimBatchId>>& outPrimBatches,
                              // I-O
                              TracerI& tracer,
                              BS::thread_pool& threadPool,
                              // Input
                              const CollapsedPrims& meshMatPrims);

MRayError ProcessUniqueSpheres(// Output
                               std::map<pxr::UsdPrim, std::vector<PrimBatchId>>& outPrimBatches,
                               // I-O
                               TracerI& tracer,
                               BS::thread_pool& threadPool,
                               // Input
                               const CollapsedPrims& meshMatPrims);