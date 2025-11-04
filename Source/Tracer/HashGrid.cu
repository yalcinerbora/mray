#include "HashGrid.h"
#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

HashGrid::HashGrid(const GPUSystem& gpuSystem)
    : gpuSystem(gpuSystem)
    , mem(gpuSystem.AllGPUs(), 64_MiB, 128_MiB, true)
{}

void HashGrid::Reset(AABB3 aabbIn, Vector3 camLocationIn,
                     uint32_t baseLevelPositionBitsIn,
                     uint32_t normalBitsIn, uint32_t maxLevelIn,
                     Float coneApertureDegrees,
                     uint32_t maxEntryCount, const GPUQueue& queue)
{
    regionAABB = aabbIn;
    camLocation = camLocationIn;
    normalBits = normalBitsIn;
    maxLevel = maxLevelIn;
    baseLevelPositionBits = baseLevelPositionBitsIn;
    coneAperture = MathConstants::DegToRadCoef<Float>() * coneApertureDegrees;

    if(maxLevel > SpatioDirCode::MaxLevel())
    {
        throw MRayError("Maximum level (which is \"{}\") for hash grid "
                        "exceeds the maximum \"{}\"!",
                        maxLevel, SpatioDirCode::MaxLevel());
    }
    if(normalBits > SpatioDirCode::NORMAL_BITS_PER_DIM)
    {
        throw MRayError("Normal bits (which is \"{}\") for hash grid "
                        "exceeds the maximum \"{}\"!",
                        normalBits, SpatioDirCode::NORMAL_BITS_PER_DIM);
    }
    if(baseLevelPositionBits > SpatioDirCode::MORTON_BITS_PER_DIM)
    {
        throw MRayError("Positional bits (which is \"{}\") for hash grid "
                        "exceeds the maximum \"{}\"!",
                        baseLevelPositionBits,
                        SpatioDirCode::MORTON_BITS_PER_DIM);
    }

    static constexpr Float BASE_LOAD_MULT = Float(1) / BASE_LOAD_FACTOR;
    uint32_t htSize = Math::NextPowerOfTwo(uint32_t(Float(maxEntryCount) * BASE_LOAD_MULT));

    MemAlloc::AllocateMultiData(Tie(dSpatialCodes, dCountBuffer),
                                mem, {htSize, 1});

    ClearAllEntries(queue);
}

void HashGrid::ClearAllEntries(const GPUQueue& queue)
{
    // Sanity check
    static constexpr auto MEMSET_FILL_PATTERN = UINT64_MAX;
    static_assert(SpatioDirCode(MEMSET_FILL_PATTERN) == HashGridView::EMPTY_VAL);
    queue.MemsetAsync(dSpatialCodes, 0xFF);
    queue.MemsetAsync(dCountBuffer, 0x00);
}

uint32_t HashGrid::UsedEntryCount(const GPUQueue& queue) const
{
    uint32_t hCount;
    queue.MemcpyAsync(Span<uint32_t>(&hCount, 1), ToConstSpan(dCountBuffer));
    queue.Barrier().Wait();
    return hCount;
}

