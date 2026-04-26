#pragma once

#include "Core/Definitions.h"
#include "Core/Vector.h"
#include "Core/MemAlloc.h"

// Sparse Volumetric Data
// Its design quite similar to the NVDB
// but it is not as generic as NVDB
//
// Data structure is optimized for read-only operation.
// It is quite static in a sense that all child nodes of a parent node
// must be contigious. Its indices are inferred from the bit mask and a
// single offset.
//
// It has single topology of 6_2_.._2 (in NVDB terms). This is chosen
// for simplicity. Base grid (2^6 portion) is dense.
//
// It is indexed data structure meaning when you sample a data
// it returns an integer that is an offset (index) of an array.
// Indices are 32-bits. This may be small but we'll see.
//
// Implementation heavily relies on popc, which does not have high througput.
// Hope it will be performant since this data structure will be used on volume rendering
// and it heavily queries the data structure. We'll see...
//
namespace VolumetricSVO
{

// Leaf node it can only be
struct alignas(16) LNode2
{
    // This is 4x4x4 leaf voxel (name convension is from NVDB,
    // 2 means the logaritmic dimension)
    //
    uint64_t valueBits;  // Bitmask: If 1, that voxel has value else 0.
    uint32_t valueStart; // Offset of these voxels. Mask and Popc to find the
                            // full offset of a certain voxel
};

struct alignas(8) INode2
{
    uint64_t nodeBits;    // Similar to the leaf struct, internal nodes also hold
                            // node bitmap. If bit is set, that node has child node
    uint64_t valueBits;   // Value bit is the same as leaf node. It also has precendence
                            // over node bit (Both should not be set at the same time but
                            // if that is the case then index of the value will be queried).
    uint32_t nodeStart;
    uint32_t valueStart;
};

struct VolGrid6_2
{
    static constexpr auto EMPTY_INDEX_VAL = uint32_t(UINT32_MAX);
    static constexpr auto BASE_BITS = uint32_t(6);
    static constexpr auto INNER_BITS = uint32_t(2);

    static constexpr auto BASE_DIM = uint32_t(1) << BASE_BITS;
    static constexpr auto BASE_DIM_CUBED = BASE_DIM * BASE_DIM * BASE_DIM;
    //
    Span<INode2> iNodes;    // Array of internal nodes ("nodeStart" variables refer to this array)
    Span<LNode2> leafs;     // Array of leafs nodes    ("nodeStart" variables refer to this array)

    uint32_t     inodeLevelCount; // This SVO has static depth, after "inodeLevelCount" iterations,
                                    // you will reach to the leaf struct (given that is available).
    Vector3      invDim;
    Vector3i     dimBits;
    // Dense Data
    // These out-of-struct so that this struct can be passed as kernel argument.
    // Compiler may put this struct on the constant memory and it has limited size
    // (64KiB).
    Span<uint64_t, 4096>           valueBits;
    Span<uint32_t, BASE_DIM_CUBED> indexList;

    // Read
    MR_PF_DECL uint32_t operator()(Vector3ui ijk) const;
    MR_PF_DECL uint32_t operator()(Vector3 uv) const;

    MR_PF_DECL Vector3  Resolution() const;
};

}

namespace VolumetricSVO
{

MR_PF_DEF
uint32_t VolGrid6_2::operator()(Vector3ui ijk) const
{
    auto GetBitRange = [&, this](uint32_t bitOffset, uint32_t bitMask) -> Vector3ui
    {
        auto baseBitOffset = Vector3ui(Math::Max(dimBits - Vector3i(bitOffset),
                                                 Vector3i::Zero()));
        Vector3ui baseDim = Vector3ui(ijk[0] >> baseBitOffset[0] & bitMask,
                                      ijk[2] >> baseBitOffset[1] & bitMask,
                                      ijk[2] >> baseBitOffset[2] & bitMask);
        return baseDim;
    };
    //
    Vector3ui baseDim = GetBitRange(BASE_BITS, 0x3F);
    uint32_t baseVoxelIndex = (baseDim[2] << 12) | (baseDim[1] << 6) | baseDim[0];
    // If has value return it
    constexpr uint32_t UINT64_BITS = sizeof(uint64_t) * CHAR_BIT;
    static_assert(Bit::PopC(UINT64_BITS) == 1,
                  "The code below does assume divisor is power of two.");
    uint32_t denseBitIndex = baseVoxelIndex & 0x3F;
    uint32_t denseArrayIndex = baseVoxelIndex >> 6;
    uint64_t hasValue = (valueBits[denseArrayIndex] >> denseBitIndex) & uint64_t(0b1);
    if(hasValue == uint64_t(0b1))
        return baseVoxelIndex;
    // If voxel is empty return empty index
    if(indexList[baseVoxelIndex] == EMPTY_INDEX_VAL)
        return EMPTY_INDEX_VAL;

    // Churn the intermediate nodes
    uint32_t curIndex = baseVoxelIndex;
    for(uint32_t i = 0; i < inodeLevelCount; i++)
    {
        Vector3ui localBits = GetBitRange(BASE_BITS + i * INNER_BITS, 0x3);
        uint32_t localI = (localBits[2] << 4) | (localBits[1] << 2) | localBits[0];
        uint64_t mask = (uint64_t(1) << localI) - uint64_t(1);
        //
        bool hasVal = ((iNodes[curIndex].valueBits >> localI) & 0x1) == 0;
        if(hasVal)
        {
            uint32_t result = iNodes[curIndex].valueStart;
            uint32_t localOffset = uint32_t(Bit::PopC(iNodes[curIndex].valueBits & mask));
            return result + localOffset;
        }
        //
        bool hasNode = ((iNodes[curIndex].nodeBits >> localI) & 0x1) == 0;
        if(!hasNode) return EMPTY_INDEX_VAL;

        uint32_t result = iNodes[curIndex].nodeStart;
        uint32_t localOffset = uint32_t(Bit::PopC(iNodes[curIndex].nodeBits & mask));
        curIndex = result + localOffset;
    }
    // Now check the leaf
    Vector3ui localBits = GetBitRange(0, 0x3);
    uint32_t localI = (localBits[2] << 4) | (localBits[1] << 2) | localBits[0];
    //
    bool hasVal = ((leafs[curIndex].valueBits >> localI) & 0x1) == 0;
    if(hasVal)
    {
        uint64_t mask = (uint64_t(1) << localI) - uint64_t(1);
        uint32_t result = leafs[curIndex].valueStart;
        uint32_t localOffset = uint32_t(Bit::PopC(leafs[curIndex].valueBits & mask));
        return result + localOffset;
    }
    return EMPTY_INDEX_VAL;
}

MR_PF_DEF
uint32_t VolGrid6_2::operator()(Vector3 uv) const
{
    Vector3ui loc = Vector3ui(uv * invDim + Vector3(0.5));
    return this->operator()(loc);
}

MR_PF_DEF
Vector3 VolGrid6_2::Resolution() const
{
    return Vector3(1) / invDim;
}

}

