#include "ReconFilter.h"

#ifdef MRAY_GPU_BACKEND_CUDA

#include <cub/block/block_reduce.cuh>

#endif

class FF
{
    static constexpr uint32_t WIDTH = 3;
};

struct RenderBufferView
{
    SoASpan<Vector3, Float> dImg;
    Vector2i imgSegmentSize;
    Vector2i imgSegmentOffset;
    Vector2i imgResolution;
};

template <class Filter, uint32_t TPB, uint32_t LOGICAL_WARP_SIZE>
MRAY_KERNEL //__launch_bounds__(TPB_X)
void KCFilterToImgWarp(MRAY_GRID_CONSTANT const RenderBufferView img,
                       // Inputs per segment
                       MRAY_GRID_CONSTANT const Span<uint32_t> dStartOffsets,
                       MRAY_GRID_CONSTANT const Span<uint32_t> dPixelIds,
                       // Inputs per thread
                       MRAY_GRID_CONSTANT const Span<uint32_t> dIndices,
                       // Inputs Accessed by SampleId
                       MRAY_GRID_CONSTANT const Span<Vector3> dValues,
                       MRAY_GRID_CONSTANT const Span<Vector2> dImgCoords,
                       // Constants
                       Float scalarMultiplier,
                       Filter FilterFunc)
{
    KernelCallParams kp;
    assert(dStartOffsets.size() == dPixelIds.size());
    static_assert(TPB % LOGICAL_WARP_SIZE == 0);

    // Some constants
    static constexpr uint32_t WARP_PER_BLOCK = TPB / LOGICAL_WARP_SIZE;
    const uint32_t totalWarpCount = WARP_PER_BLOCK * kp.gridSize;
    const uint32_t globalWarpId = kp.GlobalId() / LOGICAL_WARP_SIZE;
    const uint32_t localWarpId = kp.threadId / LOGICAL_WARP_SIZE;
    const uint32_t laneId = kp.GlobalId() % LOGICAL_WARP_SIZE;

    using WarpReduceVec4 = cub::WarpReduce<Vector4, LOGICAL_WARP_SIZE>;
    using ReduceShMem = typename WarpReduceVec4::TempStorage;
    // Per-Warp Shared Memory
    MRAY_SHARED_MEMORY Vector2ui    sSegmentRange[WARP_PER_BLOCK];
    MRAY_SHARED_MEMORY uint32_t     sResonsiblePixel[WARP_PER_BLOCK];
    MRAY_SHARED_MEMORY ReduceShMem  sReduceMem[WARP_PER_BLOCK];

    // Warp-stride loop
    uint32_t segmentCount = static_cast<uint32_t>(dStartOffsets.size());
    for(uint32_t segmentIndex = globalWarpId; segmentIndex < segmentCount;
        segmentIndex += totalWarpCount)
    {

        if(laneId == 0) sSegmentRange[localWarpId][0] = dStartOffsets[segmentIndex + 0];
        if(laneId == 1) sSegmentRange[localWarpId][1] = dStartOffsets[segmentIndex + 1];
        if(laneId == 2) sResonsiblePixel = dPixelIds[segmentIndex];

        WarpSynchronize<LOGICAL_WARP_SIZE>();




        // Next load
        WarpSynchronize<LOGICAL_WARP_SIZE>();
    }

}


template<uint32_t REGION_XY, class Filter>
MRAY_KERNEL //MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(REGION_XY* REGION_XY)
void ReconstructionFilter(// Input
                          MRAY_GRID_CONSTANT const Span<Float> dOutWeights,
                          MRAY_GRID_CONSTANT const Span<Vector3> dOutValues,
                          // Output
                          MRAY_GRID_CONSTANT Span<Span<const Vector3>> dInValues,
                          MRAY_GRID_CONSTANT Span<Span<const Vector2>> dImgCoords,
                          // Constants
                          MRAY_GRID_CONSTANT const Vector2ui regionMin,
                          MRAY_GRID_CONSTANT const Vector2ui regionMax,
                          MRAY_GRID_CONSTANT const Vector2ui resolution)
{
    //static constexpr uint32_t CELL_PER_ITERATION = 4;

    //KernelCallParams kp;
    //assert(inValues.size() = imgCoords.size());
    //uint32_t regionCount = static_cast<uint32_t>(inValues.size());
    //constexpr uint32_t PAD = Filter::WIDTH;
    //constexpr uint32_t FILTER_START = Filter::START;
    //constexpr uint32_t FILTER_END = Filter::END;
    //constexpr uint32_t PADDED_XY = REGION_XY + PAD;

    //// Staging shared memory, we filter to these, then write it
    //// to minimize atomic io.
    //MRAY_SHARED_MEMORY Vector3 stagingValues[PADDED_XY][PADDED_XY];
    //MRAY_SHARED_MEMORY Float stagingWeights[PADDED_XY][PADDED_XY];
    //MRAY_SHARED_MEMORY Span<const Vector3> currentValues;
    //MRAY_SHARED_MEMORY Span<const Vector2> currentCoords;

    //auto ClearMemory = [&]()
    //{
    //    for(uint32_t i = kp.threadId; i < PADDED_XY * PADDED_XY; i += kp.blockSize)
    //    {
    //        uint32_t x = i % PADDED_XY;
    //        uint32_t y = i / PADDED_XY;
    //        stagingValues[y][x] = Vector3::Zero();
    //        stagingWeights[y][x] = Float(0);
    //    }
    //};

    //// Block Stride Loop
    //for(uint32_t block = kp.blockId; block < regionCount; block += kp.gridSize)
    //{
    //    // Next iteration clear the staging buffer
    //    ClearMemory();
    //    if(kp.threadId == 0) currentValues = inValues[block];
    //    if(kp.threadId == 1) currentCoords = imgCoords[block];
    //    BlockSynchronize();

    //    uint32_t sampleCount = static_cast<uint32_t>(inValues.size());
    //    for(uint32_t i = kp.threadId; i < sampleCount; i += kp.blockSize)
    //    {
    //        Vector2 coords = currentCoords[i];
    //        Vector3 values = currentValues[i];

    //        uint32_t codes[CELL_PER_ITERATION];

    //        // Atomic add



    //        //fro


    //        //for(int i = 0; i < PAD; i++)
    //    }
    //}

}