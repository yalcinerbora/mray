#pragma once

#include <array>
#include <cstdint>

#include "Core/Vector.h"
#include "Core/DataStructures.h"
#include "Core/TracerI.h"
#include "Core/BitFunctions.h"

#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/prim.h>

class AttributeIndexer;
struct CollapsedPrims;
namespace BS { class thread_pool;}

using IndexTriplet = std::array<uint32_t, 3>;

struct IndexHasher
{
    // Copy paste from Tracer/Random.h
    static constexpr uint64_t PCG64(uint64_t v)
    {
        constexpr uint64_t Multiplier = 6364136223846793005ull;
        constexpr uint64_t Increment = 1442695040888963407ull;
        uint64_t s = v * Multiplier + Increment;
        uint64_t word = ((s >> ((s >> 59u) + 5u)) ^ s);
        word *= 12605985483714917081ull;
        return (word >> 43u) ^ word;
    }

    size_t operator()(const IndexTriplet& t) const
    {
        // These are random, no special meaning
        return PCG64(PCG64(PCG64(t[0] + 1) + t[1]) + t[2]);
    }
};
using IndexHashTable = std::unordered_map<IndexTriplet, uint32_t, IndexHasher>;

struct IndexLookupStrategy
{
    // Copy paste from Tracer/Random.h
    static constexpr uint64_t PCG64(uint64_t v)
    {
        constexpr uint64_t Multiplier = 6364136223846793005ull;
        constexpr uint64_t Increment = 1442695040888963407ull;
        uint64_t s = v * Multiplier + Increment;
        uint64_t word = ((s >> ((s >> 59u) + 5u)) ^ s);
        word *= 12605985483714917081ull;
        return (word >> 43u) ^ word;
    }

    static constexpr uint32_t Hash(const IndexTriplet& t)
    {
        // These are random, no special meaning
        uint64_t h = PCG64(PCG64(PCG64(t[0] + 1) + t[1]) + t[2]);

        h = Bit::FetchSubPortion(h, {0, 32}) + Bit::FetchSubPortion(h, {32, 64});

        // Skip sentinel and empty marks
        if(h == 0 || h == 1) return 2u;
        return uint32_t(h);
    }

    static constexpr bool IsSentinel(uint32_t h)
    {
        return h == 1;
    }

    static constexpr bool IsEmpty(uint32_t h)
    {
        return h == 0;
    }
};

class IndexLookupTable
{
    private:
    static constexpr std::array<size_t, 7> PRIMES =
    {
        1'000'117,
        2'000'177,
        4'000'189,
        8'000'219,
        16'000'289,
        32'000'219,
        64'000'247
        // If state is larger than 8M just double in size
    };


    private:
    std::vector<Vector<4, uint32_t>> hashes;
    std::vector<IndexTriplet> keys;
    std::vector<uint32_t> values;
    public:
    IndexLookupTable() = default;

    uint32_t maxLoadCount = 0;

    //
    void Clear()
    {
        maxLoadCount = 0;
        std::fill(hashes.begin(), hashes.end(), Vector4ui::Zero());
    }

    void Reserve(size_t requestedSize)
    {
        auto loc = std::find_if(PRIMES.cbegin(), PRIMES.cend(),
                                [requestedSize](size_t s)
        {
            return s > requestedSize;
        });
        size_t allocSize = (loc == PRIMES.cend())
            ? Math::NextMultiple(requestedSize, PRIMES.back())
            : *loc;

        hashes.resize(Math::DivideUp(allocSize, size_t(4)), Vector4ui::Zero());
        keys.resize(allocSize);
        values.resize(allocSize);
    }


    std::pair<const uint32_t*, bool> Insert(IndexTriplet key, uint32_t value)
    {
        using LT = LookupTable<IndexTriplet, uint32_t,
                               uint32_t, 4, IndexLookupStrategy>;
        //
        return LT(Span(hashes.begin(), hashes.end()),
                  Span(keys.begin(), keys.end()),
                  Span(values.begin(), values.end())).Insert(key, value);
    }
    //    static constexpr uint32_t VEC_SHIFT = std::countr_zero(4u);
    //    static constexpr uint32_t VEC_MASK = (VEC_SHIFT == 0) ? 0 : ((1u << VEC_SHIFT) - 1);

    //    using S = IndexLookupStrategy;
    //    uint32_t tableSize = static_cast<uint32_t>(keys.size());
    //    uint32_t hashPackCount = static_cast<uint32_t>(hashes.size());
    //    uint32_t hashVal = S::Hash(key);
    //    uint32_t index = uint32_t(hashVal % tableSize);

    //    uint32_t totalLoads = 0;
    //    for(uint32_t _ = 0; _ < hashPackCount; _++)
    //    {
    //        uint32_t vectorIndex = index >> VEC_SHIFT;
    //        uint32_t innerIndex = index & VEC_MASK;
    //        Vector<4, uint32_t> hashChunk = hashes[vectorIndex];

    //        totalLoads++;

    //        UNROLL_LOOP
    //        for(uint32_t i = innerIndex; i < 4; i++)
    //        {
    //            uint32_t globalIndex = (vectorIndex << VEC_SHIFT) + i;
    //            // Roll to start of the case special case
    //            // (since we are bulk reading)
    //            if(globalIndex >= tableSize) break;

    //            // Actual comparison case, if hash is equal it does not mean
    //            // keys are equal, check them only if the hashes are equal.
    //            // If true, return the old val
    //            if(hashVal == hashChunk[i] && keys[globalIndex] == key)
    //            {
    //                maxLoadCount = std::max(maxLoadCount, totalLoads);
    //                return std::pair(&values[globalIndex], false);
    //            }


    //            // If empty, this means linear probe chain is fully iterated
    //            // and we did not find the value return null
    //            if(S::IsEmpty(hashChunk[i]) || S::IsSentinel(hashChunk[i]))
    //            {
    //                hashChunk[i] = hashVal;
    //                values[globalIndex] = value;
    //                keys[globalIndex] = key;
    //                hashes[vectorIndex] = hashChunk;

    //                maxLoadCount = std::max(maxLoadCount, totalLoads);
    //                return std::pair(&values[globalIndex], true);
    //            }
    //        }
    //        index += 4 - innerIndex;
    //        index = (index >= tableSize) ? 0 : index;
    //        assert(index != hashVal % tableSize);
    //    }
    //    maxLoadCount = std::max(maxLoadCount, totalLoads);
    //    return std::pair(nullptr, false);
    //}
};

struct MeshProcessorThread
{
    template<class T>
    using StdVector2D = std::vector<std::vector<T>>;

    using SubGeomTransientData = StaticVector<TransientData, TracerConstants::MaxAttributePerGroup>;

    static const pxr::TfToken uvToken0;
    static const pxr::TfToken uvToken1;
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
    IndexLookupTable            indexLookupTable;
    //IndexHashTable              indexLookupTable;
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
                              PrimGroupId& id,
                              std::map<pxr::UsdPrim, std::vector<PrimBatchId>>& outPrimBatches,
                              // I-O
                              TracerI& tracer,
                              BS::thread_pool& threadPool,
                              // Input
                              const std::set<pxr::UsdPrim>& uniquePrims);

MRayError ProcessUniqueSpheres(// Output
                               PrimGroupId& id,
                               std::map<pxr::UsdPrim, std::vector<PrimBatchId>>& outPrimBatches,
                               // I-O
                               TracerI& tracer,
                               BS::thread_pool& threadPool,
                               // Input
                               const std::set<pxr::UsdPrim>& uniquePrims);