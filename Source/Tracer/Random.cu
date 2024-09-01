#include "Random.h"

#include <random>

#include "Device/GPUSystem.hpp"

#include <BS/BS_thread_pool.hpp>

MRAY_KERNEL
void KCGenRandomNumbersPCG32(// Output
                             Span<RandomNumber> dNumbers,
                             // I-O
                             Span<typename PermutedCG32::State> dStates,
                             // Constants
                             uint32_t dimPerGenerator)
{
    assert(dNumbers.size() == dStates.size() * dimPerGenerator);
    using State = typename PermutedCG32::State;

    uint32_t generatorCount = uint32_t(dStates.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount;
        i += kp.TotalSize())
    {
        // Get the state
        // Save the in register space so every generation do not pound the
        // global memory (Probably it will get optimized bu w/e).
        State state = dStates[i];
        PermutedCG32 rng(state);
        // PCG32 do not have concept of dimensionality (technically you can
        // hold a state for each dimension but for a path tracer it is infeasible).
        //
        // So we just generate numbers using a single state
        for(uint32_t n = 0; n < dimPerGenerator; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + dimPerGenerator * n] = rng.Next();
        }
        // Write the modified state
        dStates[i] = state;
    }
}

MRAY_KERNEL
void KCGenRandomNumbersPCG32Indirect(// Output
                                     Span<RandomNumber> dNumbers,
                                     // I-O
                                     Span<typename PermutedCG32::State> dStates,
                                     // Input
                                     Span<const RayIndex> dIndices,
                                     // Constants
                                     uint32_t dimPerGenerator)
{
    assert(dNumbers.size() == dIndices.size() * dimPerGenerator);
    using State = typename PermutedCG32::State;

    uint32_t generatorCount = uint32_t(dIndices.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount;
        i += kp.TotalSize())
    {
        // Get the state
        RayIndex index = dIndices[i];
        // Save the in register space so every generation do not pound the
        // global memory (Probably it will get optimized bu w/e).
        assert(index < dStates.size());
        State state = dStates[index];
        PermutedCG32 rng(state);
        // PCG32 do not have concept of dimensionality (technically you can
        // hold a state for each dimension but for a path tracer it is infeasible).
        //
        // So we just generate numbers using a single state
        for(uint32_t n = 0; n < dimPerGenerator; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + dimPerGenerator * n] = rng.Next();
        }
        // Write the modified state
        dStates[i] = state;
    }
}

RNGGroupIndependent::RNGGroupIndependent(Vector2ui generatorCount2D,
                                         uint64_t seed,
                                         const GPUSystem& sys,
                                         BS::thread_pool& tp)
    : mainThreadPool(tp)
    , gpuSystem(sys)
    , hostMemory(gpuSystem, true)
    , size2D(generatorCount2D)
    , deviceMemory(gpuSystem.AllGPUs(), 2_MiB, 32_MiB)
{
    size_t totalSize = size2D.Multiply();

    MemAlloc::AllocateMultiData(std::tie(hBackupStates, hMainStates),
                                hostMemory, {totalSize, totalSize});

    // These are const to catch race conditions etc.
    // TODO: Change this later
    uint32_t seed32 = static_cast<uint32_t>((seed >> 32) ^ (seed & 0xFFFFFFFF));
    const std::mt19937 rng0(seed32);
    std::mt19937 rngTemp = rng0;

    auto future0 = tp.submit_blocks(size_t(0), totalSize,
    [&rng0, this](size_t start, size_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng0;
        rngLocal.discard(start);
        for(size_t i = start; i < end; i++)
        {
            hMainStates[i] = MainRNG::GenerateState(rngLocal());
        }
    }, 4u);

    // Do discarding after issue (it should be logN for PRNGS)
    rngTemp.discard(totalSize);
    const std::mt19937 rng1 = rngTemp;

    auto future1 = tp.submit_blocks(size_t(0), totalSize,
    [&rng1, this](size_t start, size_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng1;
        rngLocal.discard(start);
        for(size_t i = start; i < end; i++)
        {
            hBackupStates[i] = BackupRNG::GenerateState(rngLocal());
        }
    }, 4u);

    future0.wait();
    future1.wait();
}

void RNGGroupIndependent::SetupDeviceRange(Vector2ui start, Vector2ui end)
{
    deviceRangeStart = start;
    deviceRangeEnd = end;
    uint32_t totalSize = (deviceRangeEnd - deviceRangeStart).Multiply();
    MemAlloc::AllocateMultiData(std::tie(dBackupStates,
                                         dMainStates),
                                deviceMemory,
                                {totalSize, totalSize});
}

void RNGGroupIndependent::CopyStatesToGPUAsync(const GPUQueue& queue)
{
    Vector2ui range = deviceRangeEnd - deviceRangeStart;
    size_t srcStride = size2D[0];
    size_t dstStride = range[0];
    size_t srcOffset = deviceRangeStart[0] + deviceRangeStart[1] * size2D[0];
    size_t srcEnd = range[0] + (deviceRangeEnd[1] - 1) * size2D[0];
    size_t srcSizeLinear = srcEnd - srcOffset;

    auto dstBackupSpan = dBackupStates;
    auto srcBackupSpan = hBackupStates.subspan(srcOffset, srcSizeLinear);
    queue.MemcpyAsync2D(dstBackupSpan, dstStride,
                        ToConstSpan(srcBackupSpan), srcStride,
                        range);

    auto dstMainSpan = dMainStates;
    auto srcMainSpan = hMainStates.subspan(srcOffset, srcSizeLinear);
    queue.MemcpyAsync2D(dstMainSpan , dstStride,
                        ToConstSpan(srcMainSpan), srcStride,
                        range);
}

void RNGGroupIndependent::CopyStatesFromGPUAsync(const GPUQueue& queue)
{
    Vector2ui range = deviceRangeEnd - deviceRangeStart;
    size_t dstStride = size2D[0];
    size_t srcStride = range[0];
    size_t dstOffset = deviceRangeStart[0] + deviceRangeStart[1] * size2D[0];
    size_t dstEnd = range[0] + (deviceRangeEnd[1] - 1) * size2D[0];
    size_t dstSizeLinear = dstEnd - dstOffset;

    auto srcBackupSpan = dBackupStates;
    auto dstBackupSpan = hBackupStates.subspan(dstOffset, dstSizeLinear);
    queue.MemcpyAsync2D(dstBackupSpan, dstStride,
                        ToConstSpan(srcBackupSpan), srcStride,
                        range);

    auto srcMainSpan = dMainStates;
    auto dstMainSpan = hMainStates.subspan(dstOffset, dstSizeLinear);
    queue.MemcpyAsync2D(dstMainSpan, dstStride,
                        ToConstSpan(srcMainSpan), srcStride,
                        range);
}

void RNGGroupIndependent::GenerateNumbers(// Output
                                          Span<RandomNumber> dNumbersOut,
                                          // Constants
                                          uint32_t dimensionCount,
                                          const GPUQueue& queue)
{
    uint32_t generatorCount = uint32_t(dMainStates.size());
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCGenRandomNumbersPCG32>
    (
        "KCGenRandomNumbersPCG32"sv,
        KernelIssueParams{.workCount = generatorCount},
        //
        dNumbersOut,
        dMainStates,
        dimensionCount
    );
}

void RNGGroupIndependent::GenerateNumbersIndirect(// Output
                                                  Span<RandomNumber> dNumbersOut,
                                                  // Input
                                                  Span<const RayIndex> dIndices,
                                                  // Constants
                                                  uint32_t dimensionCount,
                                                  const GPUQueue& queue)
{
    uint32_t generatorCount = uint32_t(dMainStates.size());
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCGenRandomNumbersPCG32Indirect>
    (
        "KCGenRandomNumbersPCG32Indirect"sv,
        KernelIssueParams{.workCount = generatorCount},
        //
        dNumbersOut,
        dMainStates,
        dIndices,
        dimensionCount
    );
}

Span<BackupRNGState> RNGGroupIndependent::GetBackupStates()
{
    return dBackupStates;
}

size_t RNGGroupIndependent::UsedGPUMemory() const
{
    return deviceMemory.Size();
}
