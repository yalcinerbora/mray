#include "Random.h"

#include <random>

#include "Device/GPUSystem.hpp"

MRAY_KERNEL
void KCGenerateRandomNumbersPCG32(// Output
                                  Span<RandomNumber> dNumbers,
                                  // I-O
                                  Span<typename PermutedCG32::State> dStates,
                                  // Input
                                  Span<const ImageCoordinate> dTileLocalPixelIds,
                                  Vector2ui tileStart,
                                  Vector2ui tileEnd,
                                  Vector2ui fullSize,
                                  uint32_t dimPerGenerator)
{
    assert(dTileLocalPixelIds.size() * dimPerGenerator == dNumbers.size());
    using State = typename PermutedCG32::State;

    uint32_t generatorCount = uint32_t(dTileLocalPixelIds.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount;
        i += kp.TotalSize())
    {
        // Get the state
        // This pixel index is relative to the tile
        Vector2ui pixelIndex = Vector2ui(dTileLocalPixelIds[i].pixelIndex);
        assert(pixelIndex < tileEnd);

        Vector2ui globalIndex = tileStart + pixelIndex;
        uint32_t globalLinearIndex = (globalIndex[1] * fullSize[0]
                                      + globalIndex[0]);

        // Save the in register space so every generation do not pound the
        // global memory (Probably it will get optimized bu w/e).
        State state = dStates[globalLinearIndex];
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

std::string_view RNGGroupIndependent::TypeName()
{
    using namespace std::string_view_literals;
    static constexpr auto Name = "Independent"sv;
    return Name;
}

RNGGroupIndependent::RNGGroupIndependent(size_t generatorCount,
                                         uint32_t seed,
                                         const GPUSystem& sys)
    : gpuSystem(sys)
    , memory(gpuSystem.AllGPUs(), 2_MiB, 64_MiB, false)
{
    MemAlloc::AllocateMultiData(std::tie(dBackupStates, dMainStates),
                                memory,
                                {generatorCount, generatorCount});

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);

    std::mt19937 rng(seed);
    std::vector<MainRNGState> hMainRNGStates(generatorCount);

    for(auto& state : hMainRNGStates)
        state = MainRNG::GenerateState(rng());
    Span<const MainRNGState> hMainRNGStateSpan(hMainRNGStates);
    queue.MemcpyAsync(dMainStates, hMainRNGStateSpan);

    std::vector<BackupRNGState> hBackupRNGStates(generatorCount);
    for(auto& state : hBackupRNGStates)
        state = BackupRNG::GenerateState(rng());
    Span<const BackupRNGState> hBackupRNGStateSpan(hMainRNGStates);
    queue.MemcpyAsync(dBackupStates, hBackupRNGStateSpan);

    queue.Barrier().Wait();
}

void RNGGroupIndependent::GenerateNumbers(Span<RandomNumber> numbersOut,
                                          Span<const ImageCoordinate> dPixelCoords,
                                          Vector2ui tileStart,
                                          Vector2ui tileEnd,
                                          Vector2ui fullSize,
                                          uint32_t dimensionCount,
                                          const GPUQueue& queue)
{
    uint32_t generatorCount = uint32_t(dPixelCoords.size());
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCGenerateRandomNumbersPCG32>
    (
        "KCGenerateRandomNumbersPCG32"sv,
        KernelIssueParams{.workCount = generatorCount},
        //
        numbersOut,
        dMainStates,
        dPixelCoords,
        tileStart,
        tileEnd,
        fullSize,
        dimensionCount
    );
}

size_t RNGGroupIndependent::UsedGPUMemory() const
{
    return memory.Size();
}