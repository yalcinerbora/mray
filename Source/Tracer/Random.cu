#include "Random.h"

#include <random>

#include "Device/GPUSystem.hpp"

#include "Core/ThreadPool.h"

MRAY_KERNEL
void KCGenRandomNumbersPCG32(// Output
                             Span<RandomNumber> dNumbers,
                             // I-O
                             Span<typename PermutedCG32::State> dStates,
                             // Constants
                             uint32_t dimPerGenerator)
{
    assert(dStates.size() * dimPerGenerator <= dNumbers.size());

    uint32_t generatorCount = uint32_t(dStates.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount;
        i += kp.TotalSize())
    {
        // Generate RNG, it automatically saves the state in register-space,
        // writes back on destruction
        PermutedCG32 rng(dStates[i]);
        // PCG32 do not have concept of dimensionality (technically you can
        // hold a state for each dimension but for a path tracer it is infeasible).
        //
        // So we just generate numbers using a single state
        for(uint32_t n = 0; n < dimPerGenerator; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + generatorCount * n] = rng.Next();
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

    uint32_t generatorCount = uint32_t(dIndices.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount;
        i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        assert(index < dStates.size());
        // Generate RNG, it automatically saves the state in register-space,
        // writes back on destruction
        PermutedCG32 rng(dStates[index]);
        // PCG32 do not have concept of dimensionality (technically you can
        // hold a state for each dimension but for a path tracer it is infeasible).
        //
        // So we just generate numbers using a single state
        for(uint32_t n = 0; n < dimPerGenerator; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + generatorCount * n] = rng.Next();
        }
    }
}

RNGGroupIndependent::RNGGroupIndependent(uint32_t genCount,
                                         uint64_t seed,
                                         const GPUSystem& sys,
                                         ThreadPool& tp)
    : mainThreadPool(tp)
    , gpuSystem(sys)
    , generatorCount(genCount)
    , currentRange(0, generatorCount)
    , deviceMem(gpuSystem.AllGPUs(), 2_MiB, 32_MiB)
{
    MemAlloc::AllocateMultiData(Tie(dBackupStates, dMainStates),
                                deviceMem,
                                {generatorCount, generatorCount});

    // These are const to catch race conditions etc.
    // TODO: Change this later
    uint32_t seed32 = static_cast<uint32_t>((seed >> 32) ^ (seed & 0xFFFFFFFF));
    const std::mt19937 rng0(seed32);
    std::mt19937 rngTemp = rng0;

    std::vector<MainRNGState> hMainStates(generatorCount);
    auto future0 = tp.SubmitBlocks(generatorCount,
    [&rng0, &hMainStates](size_t start, size_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng0;
        rngLocal.discard(start);
        for(size_t i = start; i < end; i++)
        {
            auto xi = rngLocal();
            hMainStates[i] = MainRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future0.WaitAll();

    // Do discarding after issue (it should be logN for PRNGS)
    rngTemp.discard(generatorCount);
    const std::mt19937 rng1 = rngTemp;

    std::vector<BackupRNGState> hBackupStates(generatorCount);
    auto future1 = tp.SubmitBlocks(generatorCount,
    [&rng1, &hBackupStates](size_t start, size_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng1;
        rngLocal.discard(start);
        for(size_t i = start; i < end; i++)
        {
            auto xi = rngLocal();
            hBackupStates[i] = BackupRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future1.WaitAll();

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
                                          Vector2ui dimensionRange,
                                          const GPUQueue& queue) const
{
    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
    uint32_t dimensionCount = dimensionRange[1] - dimensionRange[0];
    uint32_t localGenCount = currentRange[1] - currentRange[0];
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32>
    (
        "KCGenRandomNumbersPCG32"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStates.subspan(currentRange[0], localGenCount),
        dimensionCount
    );
}

void RNGGroupIndependent::GenerateNumbersIndirect(// Output
                                                  Span<RandomNumber> dNumbersOut,
                                                  // Input
                                                  Span<const RayIndex> dIndices,
                                                  // Constants
                                                  Vector2ui dimensionRange,
                                                  const GPUQueue& queue) const
{
    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
    uint32_t dimensionCount = dimensionRange[1] - dimensionRange[0];
    uint32_t localGenCount = currentRange[1] - currentRange[0];
    uint32_t usedGenCount = static_cast<uint32_t>(dIndices.size());
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32Indirect>
    (
        "KCGenRandomNumbersPCG32Indirect"sv,
        DeviceWorkIssueParams{.workCount = usedGenCount},
        //
        dNumbersOut,
        dMainStates.subspan(currentRange[0], localGenCount),
        dIndices,
        dimensionCount
    );
}

void RNGGroupIndependent::GenerateNumbersIndirect(// Output
                                                  Span<RandomNumber> dNumbersOut,
                                                  // Input
                                                  Span<const RayIndex> dIndices,
                                                  Span<const uint32_t>,
                                                  // Constants
                                                  uint32_t dimensionCount,
                                                  const GPUQueue& queue) const
{
    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
    uint32_t localGenCount = currentRange[1] - currentRange[0];
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32Indirect>
        (
            "KCGenRandomNumbersPCG32Indirect"sv,
            DeviceWorkIssueParams{.workCount = localGenCount},
            //
            dNumbersOut,
            dMainStates.subspan(currentRange[0], localGenCount),
            dIndices,
            dimensionCount
        );
}

Span<BackupRNGState> RNGGroupIndependent::GetBackupStates()
{
    return dBackupStates.subspan(currentRange[0],
                                 currentRange[1] - currentRange[0]);
}

size_t RNGGroupIndependent::GPUMemoryUsage() const
{
    return deviceMem.Size();
}
