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
    assert(dStates.size() * dimPerGenerator == dNumbers.size());
    using State = typename PermutedCG32::State;

    uint32_t generatorCount = uint32_t(dStates.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount;
        i += kp.TotalSize())
    {
        // Get the state
        // Save the in register space so every generation do not pound the
        // global memory (Probably it will get optimized bu w/e).
        PermutedCG32 rng(dStates[i]);
        // PCG32 do not have concept of dimensionality (technically you can
        // hold a state for each dimension but for a path tracer it is infeasible).
        //
        // So we just generate numbers using a single state
        RandomNumber rn[2];
        for(uint32_t n = 0; n < dimPerGenerator; n++)
        {
            // Write in strided fashion to coalesce mem
            rn[n] = rng.Next();
            dNumbers[i + generatorCount * n] = rn[n];
        }
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

RNGGroupIndependent::RNGGroupIndependent(uint32_t genCount,
                                         uint64_t seed,
                                         const GPUSystem& sys,
                                         BS::thread_pool& tp)
    : mainThreadPool(tp)
    , gpuSystem(sys)
    , generatorCount(genCount)
    , currentRange(0, generatorCount)
    , deviceMem(gpuSystem.AllGPUs(), 2_MiB, 32_MiB)
{
    MemAlloc::AllocateMultiData(std::tie(dBackupStates, dMainStates),
                                deviceMem,
                                {generatorCount, generatorCount});

    // These are const to catch race conditions etc.
    // TODO: Change this later
    uint32_t seed32 = static_cast<uint32_t>((seed >> 32) ^ (seed & 0xFFFFFFFF));
    const std::mt19937 rng0(seed32);
    std::mt19937 rngTemp = rng0;

    std::vector<MainRNGState> hMainStates(generatorCount);
    auto future0 = tp.submit_blocks(size_t(0), size_t(generatorCount),
    [&rng0, &hMainStates](size_t start, size_t end)
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
    rngTemp.discard(genCount);
    const std::mt19937 rng1 = rngTemp;

    std::vector<BackupRNGState> hBackupStates(generatorCount);
    auto future1 = tp.submit_blocks(size_t(0), size_t(generatorCount),
    [&rng1, &hBackupStates](size_t start, size_t end)
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

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    queue.MemcpyAsync(dBackupStates, Span<const BackupRNGState>(hBackupStates));
    queue.MemcpyAsync(dMainStates, Span<const MainRNGState>(hMainStates));
    queue.Barrier().Wait();
}

void RNGGroupIndependent::SetupRange(Vector2ui range)
{
    currentRange = range;
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
        dMainStates.subspan(currentRange[0],
                            currentRange[1] - currentRange[0]),
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
        dMainStates.subspan(currentRange[0],
                            currentRange[1] - currentRange[0]),
        dIndices,
        dimensionCount
    );
}

Span<BackupRNGState> RNGGroupIndependent::GetBackupStates()
{
    return dBackupStates.subspan(currentRange[0],
                                 currentRange[1] - currentRange[0]);
}

size_t RNGGroupIndependent::UsedGPUMemory() const
{
    return deviceMem.Size();
}
