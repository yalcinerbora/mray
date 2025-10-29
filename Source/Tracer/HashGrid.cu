#include "HashGrid.h"
#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgReduce.h"

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

    using DeviceAlgorithms::TransformReduceTMSize;
    size_t tempMemSize = TransformReduceTMSize<uint32_t, SpatioDirCode>(htSize, queue);
    MemAlloc::AllocateMultiData(Tie(dSpatialCodes, dTransformReduceTempMem, dCountBuffer),
                                mem, {htSize, tempMemSize, 1});

    // Lets do some sanity check here to catch errors
    static constexpr auto MEMSET_FILL_PATTERN = UINT64_MAX;
    static_assert(SpatioDirCode(MEMSET_FILL_PATTERN) == HashGridView::EMPTY_VAL);
    queue.MemsetAsync(dSpatialCodes, 0xFF);
    queue.Barrier().Wait();
}

uint32_t HashGrid::CalculateUsedGridCount(const GPUQueue& queue) const
{
    DeviceAlgorithms::TransformReduce
    (
        Span<uint32_t, 1>(dCountBuffer),
        dTransformReduceTempMem, ToConstSpan(dSpatialCodes),
        uint32_t(0), queue,
        [] MRAY_GPU(uint32_t a, uint32_t b) -> uint32_t
        {
            return a + b;
        },
        [] MRAY_GPU(SpatioDirCode code)->uint32_t
        {
            static constexpr auto E_VAL = HashGridView::EMPTY_VAL;
            static constexpr auto S_VAL = HashGridView::SENTINEL_VAL;
            return (code == S_VAL || code == E_VAL) ? 0 : 1;
        }
    );

    uint32_t hCount = 0;
    queue.MemcpyAsync(Span<uint32_t>(&hCount, 1), ToConstSpan(dCountBuffer));
    queue.Barrier().Wait();

    return hCount;
}

