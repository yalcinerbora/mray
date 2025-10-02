#include "Random.h"

#include <random>

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"

#include "Core/ThreadPool.h"
#include "Core/TracerI.h"
#include "Core/GraphicsFunctions.h"

namespace ZSobolDetail
{
    static constexpr size_t SOBOL_MATRIX_SIZE = 52;

    // From
    // https://github.com/lgruen/sobol/blob/main/single-precision/sobol.cpp
    static constexpr std::array SOBOL_32_JOE_KUO_DIM_0 =
    {
        0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 0x08000000u, 0x04000000u,
        0x02000000u, 0x01000000u, 0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
        0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u, 0x00008000u, 0x00004000u,
        0x00002000u, 0x00001000u, 0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
        0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u, 0x00000008u, 0x00000004u,
        0x00000002u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
    };

    static constexpr std::array SOBOL_32_JOE_KUO_DIM_1 =
    {
        0x80000000u, 0xC0000000u, 0xA0000000u, 0xF0000000u, 0x88000000u, 0xCC000000u,
        0xAA000000u, 0xFF000000u, 0x80800000u, 0xC0C00000u, 0xA0A00000u, 0xF0F00000u,
        0x88880000u, 0xCCCC0000u, 0xAAAA0000u, 0xFFFF0000u, 0x80008000u, 0xC000C000u,
        0xA000A000u, 0xF000F000u, 0x88008800u, 0xCC00CC00u, 0xAA00AA00u, 0xFF00FF00u,
        0x80808080u, 0xC0C0C0C0u, 0xA0A0A0A0u, 0xF0F0F0F0u, 0x88888888u, 0xCCCCCCCCu,
        0xAAAAAAAAu, 0xFFFFFFFFu, 0x80000000u, 0xC0000000u, 0xA0000000u, 0xF0000000u,
        0x88000000u, 0xCC000000u, 0xAA000000u, 0xFF000000u, 0x80800000u, 0xC0C00000u,
        0xA0A00000u, 0xF0F00000u, 0x88880000u, 0xCCCC0000u, 0xAAAA0000u, 0xFFFF0000u,
        0x80008000u, 0xC000C000u, 0xA000A000u, 0xF000F000u
    };
    static_assert(SOBOL_32_JOE_KUO_DIM_1.size() == SOBOL_MATRIX_SIZE &&
                  SOBOL_32_JOE_KUO_DIM_1.size() == SOBOL_MATRIX_SIZE);

    template<std::array<uint32_t, SOBOL_MATRIX_SIZE>>
    MR_PF_DEF uint32_t SampleSobol32(uint64_t a);

    // Utility
    MR_PF_DECL uint64_t MixBits(uint64_t);
    // Scramblers
    MR_PF_DECL uint32_t ScambleOwen(uint32_t v, uint32_t seed);
    MR_PF_DECL uint32_t ScambleOwenFast(uint32_t v, uint32_t seed);

    class ZSobol
    {
        const GlobalState&  globalState;
        uint32_t            seed;
        int32_t             nBase4Digits;
        uint32_t            log2SPP;
        uint64_t            mortonIndex;

        private:
        MR_HF_DECL uint64_t SampleIndex(uint32_t dim) const;
        MR_PF_DECL uint32_t SampleSobol32(uint64_t a, uint32_t dim) const;

        public:
        // Constructors & Destructor
        MR_PF_DECL_V ZSobol(const LocalState&, const GlobalState&);

        MR_HF_DECL
        uint32_t    Next(uint32_t dim) const;
    };
}

// More or less 1-1 of the PBRT
// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/hash.h#L70
// Which is from this blog post.
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
MR_PF_DEF
uint64_t ZSobolDetail::MixBits(uint64_t v)
{
    v ^= (v >> 31ull);
    v *= 0x7FB5D329728EA185ull;
    v ^= (v >> 27ull);
    v *= 0x81DADEF4BC2DD44Dull;
    v ^= (v >> 33ull);
    return v;
};

template<std::array<uint32_t, ZSobolDetail::SOBOL_MATRIX_SIZE> SobolArray>
MR_PF_DEF uint32_t ZSobolDetail::SampleSobol32(uint64_t a)
{
    // This code is shelved, since the code below generates exactly the same
    // assembly on NVCC.
    //
    //#define EXPAND_JOE_KUO_DIM(i)               \
    //    if(a == 0) return result;               \
    //    if(a & 0x1) result ^= SobolArray[i];    \
    //    a >>= 1                                 \
    //uint32_t result = 0;
    //EXPAND_JOE_KUO_DIM(0); EXPAND_JOE_KUO_DIM(1); EXPAND_JOE_KUO_DIM(2);
    //EXPAND_JOE_KUO_DIM(3); EXPAND_JOE_KUO_DIM(4); EXPAND_JOE_KUO_DIM(5);
    //EXPAND_JOE_KUO_DIM(6); EXPAND_JOE_KUO_DIM(7); EXPAND_JOE_KUO_DIM(8);
    //EXPAND_JOE_KUO_DIM(9); EXPAND_JOE_KUO_DIM(10); EXPAND_JOE_KUO_DIM(11);
    //EXPAND_JOE_KUO_DIM(12); EXPAND_JOE_KUO_DIM(13); EXPAND_JOE_KUO_DIM(14);
    //EXPAND_JOE_KUO_DIM(15); EXPAND_JOE_KUO_DIM(16); EXPAND_JOE_KUO_DIM(17);
    //EXPAND_JOE_KUO_DIM(18); EXPAND_JOE_KUO_DIM(19); EXPAND_JOE_KUO_DIM(20);
    //EXPAND_JOE_KUO_DIM(21); EXPAND_JOE_KUO_DIM(22); EXPAND_JOE_KUO_DIM(23);
    //EXPAND_JOE_KUO_DIM(24); EXPAND_JOE_KUO_DIM(25); EXPAND_JOE_KUO_DIM(26);
    //EXPAND_JOE_KUO_DIM(27); EXPAND_JOE_KUO_DIM(28); EXPAND_JOE_KUO_DIM(29);
    //EXPAND_JOE_KUO_DIM(30); EXPAND_JOE_KUO_DIM(31); EXPAND_JOE_KUO_DIM(32);
    //EXPAND_JOE_KUO_DIM(33); EXPAND_JOE_KUO_DIM(34); EXPAND_JOE_KUO_DIM(35);
    //EXPAND_JOE_KUO_DIM(36); EXPAND_JOE_KUO_DIM(37); EXPAND_JOE_KUO_DIM(38);
    //EXPAND_JOE_KUO_DIM(39); EXPAND_JOE_KUO_DIM(40); EXPAND_JOE_KUO_DIM(41);
    //EXPAND_JOE_KUO_DIM(42); EXPAND_JOE_KUO_DIM(43); EXPAND_JOE_KUO_DIM(44);
    //EXPAND_JOE_KUO_DIM(45); EXPAND_JOE_KUO_DIM(46); EXPAND_JOE_KUO_DIM(47);
    //EXPAND_JOE_KUO_DIM(48); EXPAND_JOE_KUO_DIM(49); EXPAND_JOE_KUO_DIM(50);
    //EXPAND_JOE_KUO_DIM(51);
    //return result;
    //#undef EXPAND_JOE_KUO_DIM

    // Let compiler decide to embed the values as immediates
    uint32_t result = 0;
    MRAY_UNROLL_LOOP
    for(uint32_t i = 0; i < SOBOL_MATRIX_SIZE; i++)
    {
        if(a == 0) return result;
        if(a & 0x1) result ^= SobolArray[i];
        a >>= 1;
    }
    return result;
}

MR_PF_DEF
uint32_t ZSobolDetail::ScambleOwen(uint32_t v, uint32_t seed)
{
    if(seed & 1) v ^= 1u << 31u;
    //
    MRAY_UNROLL_LOOP
    for(uint32_t b = 1; b < 32u; b++)
    {
        uint32_t mask = (~0u) << (32u - b);
        uint32_t mixResult = uint32_t(MixBits((v & mask) ^ seed));
        if(mixResult & (1u << b))
            v ^= 1u << (31u - b);
    }
    return v;
}

MR_PF_DEF
uint32_t ZSobolDetail::ScambleOwenFast(uint32_t v, uint32_t seed)
{
    v = Bit::BitReverse(v);
    v ^= v * 0x3A20ADEAu;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526C56u;
    v ^= v * 0x53A22864u;
    return v;
}

MR_HF_DEF
uint64_t ZSobolDetail::ZSobol::SampleIndex(uint32_t dimension) const
{
    constexpr auto GetPermutation = [](uint32_t i, uint32_t d) -> uint32_t
    {
        // Unlike the PBRT book, we do not use static array of chars
        // GPU compiler hates these, and never lifts these to register space
        // and stays in local memory (especially when data type is char etc.).
        //
        // So I've come up with this compressed LUT that holds 4 64-bit integers
        // These should be put directly into the instructions as immediates. (hopefully)
        //
        // We could've use 3 64-bit integers but then fetching those would require
        // extra instructions.
        #define COMPOSE_48x2(...) Bit::Compose      \
            <                                       \
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
                16                                  \
            >                                       \
            (__VA_ARGS__, 0u)

        // Sanity check, we use "ull" literal suffix to deduce the "Compose"
        // function's arguments, so if compiler changes this should fail.
        static_assert(std::is_same_v<decltype(0ull), uint64_t>);

        // After multiple implementations, this seems to be the
        // fastest (althought slightly)
        static constexpr std::array<uint64_t, 4> TABLE =
        {
            COMPOSE_48x2(0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
                         1ull, 1ull, 1ull, 1ull, 1ull, 1ull,
                         2ull, 2ull, 2ull, 2ull, 2ull, 2ull,
                         3ull, 3ull, 3ull, 3ull, 3ull, 3ull),
            //
            COMPOSE_48x2(1ull, 1ull, 2ull, 2ull, 3ull, 3ull,
                         0ull, 0ull, 2ull, 2ull, 3ull, 3ull,
                         1ull, 1ull, 0ull, 0ull, 3ull, 3ull,
                         1ull, 1ull, 2ull, 2ull, 0ull, 0ull),
            //
            COMPOSE_48x2(2ull, 3ull, 1ull, 3ull, 2ull, 1ull,
                         2ull, 3ull, 0ull, 3ull, 2ull, 0ull,
                         0ull, 3ull, 1ull, 3ull, 0ull, 1ull,
                         2ull, 0ull, 1ull, 0ull, 2ull, 1ull),
            //
            COMPOSE_48x2(3ull, 2ull, 3ull, 1ull, 1ull, 2ull,
                         3ull, 2ull, 3ull, 0ull, 0ull, 2ull,
                         3ull, 0ull, 3ull, 1ull, 1ull, 0ull,
                         0ull, 2ull, 0ull, 1ull, 1ull, 2ull)
        };
        #undef COMPOSE_48x2
        //
        assert(d < 4 && i < 24);
        return uint32_t(Bit::FetchSubPortion(TABLE[d], {i * 2, i * 2 + 2}));
    };

    const uint32_t dimMixer = 0x55555555u * dimension;
    uint64_t sampleIndex = 0;
    // Apply random permutations to full base-4 digits
    bool isPow2Samples = log2SPP & 1;
    int32_t lastDigit = isPow2Samples ? 1 : 0;
    for(int32_t i = nBase4Digits - 1; i >= lastDigit; i--)
    {
        uint32_t digitShift = uint32_t(2 * i - (isPow2Samples ? 1 : 0));
        uint32_t digit = (mortonIndex >> digitShift) & 0b11u;

        uint64_t higherDigits = mortonIndex >> (digitShift + 2u);
        uint32_t p = (MixBits(higherDigits ^ dimMixer) >> 24u) % 24u;

        digit = GetPermutation(p, digit);
        sampleIndex |= uint64_t(digit) << digitShift;
    }
    //
    if(isPow2Samples)
    {
        uint32_t digit = uint32_t(mortonIndex & 1u);
        sampleIndex |= digit ^ (MixBits((mortonIndex >> 1u) ^ dimMixer) & 1u);
    }
    return sampleIndex;
}

MR_PF_DEF
uint32_t ZSobolDetail::ZSobol::SampleSobol32(uint64_t a, uint32_t dim) const
{
    assert(dim < 2);
    if(dim == 0)
        return ZSobolDetail::SampleSobol32<SOBOL_32_JOE_KUO_DIM_0>(a);
    else
        return ZSobolDetail::SampleSobol32<SOBOL_32_JOE_KUO_DIM_1>(a);
}

MR_PF_DEF_V
ZSobolDetail::ZSobol::ZSobol(const LocalState& ls,
                             const GlobalState& gs)
    : globalState(gs)
    , seed(ls.seed)
{
    // Determine the current sample rotation from sampleId
    // Loop for worst case, maxSPP was 1 and we did ~4billion samples
    uint32_t sppLocalSampleIndex = ls.sampleIndex;
    uint32_t sppLocalMaxSPP = gs.initialMaxSPP;

    MRAY_UNROLL_LOOP_N(8)
    for(uint32_t i = 0; i < 32; i++)
    {
        uint32_t curSize = gs.initialMaxSPP << i;
        //
        if(sppLocalSampleIndex < curSize) break;
        //
        sppLocalSampleIndex -= curSize;
        sppLocalMaxSPP <<= 1;
    }
    //
    log2SPP = Bit::RequiredBitsToRepresent(sppLocalMaxSPP) - 1;
    uint32_t log4SPP = Math::DivideUp(log2SPP, 2u);
    nBase4Digits = int32_t(gs.resMaxBits + log4SPP);

    mortonIndex = (ls.pixelMortonCode << log2SPP) | sppLocalSampleIndex;
}

MR_HF_DEF
uint32_t ZSobolDetail::ZSobol::Next(uint32_t dim) const
{
    uint32_t dimU = dim & (0xFFFFFFFE);
    uint32_t dimL = dim & (0x00000001);
    uint64_t sampleIndex = SampleIndex(dimU);
    uint32_t sample = SampleSobol32(sampleIndex, dimL);

    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    //sample = ScambleOwenFast(sample, uint32_t(sampleHash));
    return sample;
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersPCG32(// Output
                             MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                             // I-O
                             MRAY_GRID_CONSTANT const Span<typename PermutedCG32::State> dStates,
                             // Constants
                             MRAY_GRID_CONSTANT const uint32_t dimPerGenerator)
{
    assert(dStates.size() * dimPerGenerator <= dNumbers.size());

    uint32_t generatorCount = uint32_t(dStates.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersPCG32Indirect(// Output
                                     MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                                     // I-O
                                     MRAY_GRID_CONSTANT const Span<typename PermutedCG32::State> dStates,
                                     // Input
                                     MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                     // Constants
                                     MRAY_GRID_CONSTANT const uint32_t dimPerGenerator)
{
    assert(dNumbers.size() == dIndices.size() * dimPerGenerator);

    uint32_t generatorCount = uint32_t(dIndices.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersZSobol(// Output
                              MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                              // I-O
                              MRAY_GRID_CONSTANT const Span<const ZSobolDetail::LocalState> dStates,
                              // Constants
                              MRAY_GRID_CONSTANT const Vector2ui dimRange,
                              MRAY_GRID_CONSTANT const ZSobolDetail::GlobalState globalState)
{
    assert(dStates.size() * (dimRange[1] - dimRange[0]) <= dNumbers.size());

    KernelCallParams kp;
    uint32_t generatorCount = uint32_t(dStates.size());
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
    {
        ZSobolDetail::ZSobol rng(dStates[i], globalState);

        uint32_t dimCount = dimRange[1] - dimRange[0];
        for(uint32_t n = 0; n < dimCount; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + generatorCount * n] = rng.Next(dimRange[0] + n);
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersZSobolIndirect(// Output
                                      MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                                      // I-O
                                      MRAY_GRID_CONSTANT const Span<ZSobolDetail::LocalState> dStates,
                                      // Input
                                      MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                      // Constants
                                      MRAY_GRID_CONSTANT const Vector2ui dimRange,
                                      MRAY_GRID_CONSTANT const ZSobolDetail::GlobalState globalState)
{
    KernelCallParams kp;
    assert(dNumbers.size() == dIndices.size() * (dimRange[1] - dimRange[0]));

    uint32_t generatorCount = uint32_t(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        assert(index < dStates.size());

        ZSobolDetail::ZSobol rng(dStates[index], globalState);
        //
        uint32_t dimCount = dimRange[1] - dimRange[0];
        for(uint32_t n = 0; n < dimCount; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + generatorCount * n] = rng.Next(dimRange[0] + n);
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersZSobolIndirectDynamicDim(// Output
                                                MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                                                // I-O
                                                MRAY_GRID_CONSTANT const Span<ZSobolDetail::LocalState> dStates,
                                                // Input
                                                MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                                MRAY_GRID_CONSTANT const Span<const uint32_t> dCurrentDimensions,
                                                // Constants
                                                MRAY_GRID_CONSTANT const uint32_t dimPerGenerator,
                                                MRAY_GRID_CONSTANT const ZSobolDetail::GlobalState globalState)
{
    assert(dNumbers.size() == dIndices.size() * dimPerGenerator);
    KernelCallParams kp;
    uint32_t generatorCount = uint32_t(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        assert(index < dStates.size());

        ZSobolDetail::ZSobol rng(dStates[index], globalState);
        //
        uint32_t dimStart = dCurrentDimensions[index];
        for(uint32_t n = 0; n < dimPerGenerator; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + generatorCount * n] = rng.Next(dimStart + n);
        }
    }
}

RNGGroupIndependent::RNGGroupIndependent(const RenderImageParams& rip,
                                         Vector2ui maxGPUPresentRNGCount,
                                         uint32_t, uint64_t seed,
                                         const GPUSystem& sys,
                                         ThreadPool& tp)
    : mainThreadPool(tp)
    , gpuSystem(sys)
    , generatorCount(rip.regionMax - rip.regionMin)
    , currentRange{Vector2ui::Zero(), Vector2ui::Zero()}
    , hostMem(gpuSystem)
    , deviceMem(gpuSystem.AllGPUs(), 2_MiB, 32_MiB)
{
    size_t gpuRNGCountMax = maxGPUPresentRNGCount.Multiply();
    MemAlloc::AllocateMultiData(Tie(dBackupStatesLocal, dMainStatesLocal),
                                deviceMem,
                                {gpuRNGCountMax , gpuRNGCountMax});

    size_t totalRNGCount = generatorCount.Multiply();
    MemAlloc::AllocateMultiData(Tie(hBackupStatesAll, hMainStatesAll),
                                hostMem,
                                {totalRNGCount , totalRNGCount});

    // These are const to catch race conditions etc.
    // TODO: Change this later
    uint32_t seed32 = static_cast<uint32_t>((seed >> 32) ^ (seed & 0xFFFFFFFF));
    const std::mt19937 rng0(seed32);
    std::mt19937 rngTemp = rng0;

    auto hMainStates = hMainStatesAll;
    auto future0 = tp.SubmitBlocks(uint32_t(totalRNGCount),
    [&rng0, hMainStates](uint32_t start, uint32_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng0;
        rngLocal.discard(start);
        for(uint32_t i = start; i < end; i++)
        {
            auto xi = rngLocal();
            hMainStates[i] = MainRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future0.WaitAll();

    // Do discarding after issue (it should be logN for PRNGS)
    rngTemp.discard(totalRNGCount);
    const std::mt19937 rng1 = rngTemp;

    auto hBackupStates = hBackupStatesAll;
    auto future1 = tp.SubmitBlocks(uint32_t(totalRNGCount),
    [&rng1, hBackupStates](uint32_t start, uint32_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng1;
        rngLocal.discard(start);
        for(uint32_t i = start; i < end; i++)
        {
            auto xi = rngLocal();
            hBackupStates[i] = BackupRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future1.WaitAll();
}

void RNGGroupIndependent::SetupRange(Vector2ui rangeStart,
                                     Vector2ui rangeEnd,
                                     const GPUQueue& queue)
{
    // Skip if exactly the same range is requested
    if(rangeStart == currentRange[0] &&
       rangeEnd == currentRange[1]) return;

    auto GenMemcpy2DParams = [&, this](const std::array<Vector2ui, 2>& rangeIn)
    {
        size_t gpuStride = (rangeIn[1] - rangeIn[0])[0];
        size_t hostStride = generatorCount[0];
        //
        size_t gpuOffset = 0;
        size_t hostOffset = rangeIn[0][1] * hostStride + rangeIn[0][0];
        return std::array{gpuStride, gpuOffset, hostStride, hostOffset};
    };

    // Copy churned rngs back to the host
    if(currentRange[0] != Vector2ui::Zero() ||
       currentRange[1] != Vector2ui::Zero())
    {

        auto [gpuStride, gpuOffset,
              hostStride, hostOffset] = GenMemcpy2DParams(currentRange);

        queue.MemcpyAsync2D(hBackupStatesAll.subspan(hostOffset), hostStride,
                            ToConstSpan(dBackupStatesLocal), gpuStride,
                            (currentRange[1] - currentRange[0]));
        queue.MemcpyAsync2D(hMainStatesAll.subspan(hostOffset), hostStride,
                            ToConstSpan(dMainStatesLocal), gpuStride,
                            (currentRange[1] - currentRange[0]));
    }
    //
    {
        currentRange = {rangeStart, rangeEnd};
        auto [gpuStride, gpuOffset,
              hostStride, hostOffset] = GenMemcpy2DParams(currentRange);

        queue.MemcpyAsync2D(dBackupStatesLocal, gpuStride,
                            ToConstSpan(hBackupStatesAll.subspan(hostOffset)),
                            hostStride, (currentRange[1] - currentRange[0]));
        queue.MemcpyAsync2D(dMainStatesLocal, gpuStride,
                            ToConstSpan(hMainStatesAll.subspan(hostOffset)),
                            hostStride, (currentRange[1] - currentRange[0]));
    }
}

void RNGGroupIndependent::GenerateNumbers(// Output
                                          Span<RandomNumber> dNumbersOut,
                                          // Constants
                                          Vector2ui dimensionRange,
                                          const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
    uint32_t dimensionCount = dimensionRange[1] - dimensionRange[0];
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32>
    (
        "KCGenRandomNumbersPCG32"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
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
    assert(currentRange[1] != Vector2ui::Zero());

    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
    uint32_t dimensionCount = dimensionRange[1] - dimensionRange[0];
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    uint32_t usedGenCount = static_cast<uint32_t>(dIndices.size());
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32Indirect>
    (
        "KCGenRandomNumbersPCG32Indirect"sv,
        DeviceWorkIssueParams{.workCount = usedGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
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
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32Indirect>
    (
        "KCGenRandomNumbersPCG32Indirect"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dIndices,
        dimensionCount
    );
}

void RNGGroupIndependent::IncrementSampleId(const GPUQueue&) const
{
    // Independent sampler does not have a notion of dimensions so
    // do nothing
}

void RNGGroupIndependent::IncrementSampleIdIndirect(Span<const RayIndex>,
                                                    const GPUQueue&) const
{
    // Independent sampler does not have a notion of dimensions so
    // do nothing
}

Span<BackupRNGState> RNGGroupIndependent::GetBackupStates()
{
    assert(currentRange[1] != Vector2ui::Zero());
    size_t localRNGCount = (currentRange[1] - currentRange[0]).Multiply();
    return dBackupStatesLocal.subspan(0u, localRNGCount);
}

size_t RNGGroupIndependent::GPUMemoryUsage() const
{
    return deviceMem.Size();
}

RNGGroupZSobol::RNGGroupZSobol(const RenderImageParams& rip,
                               Vector2ui maxGPUPresentRNGCount,
                               uint32_t initialMaxSampleCount,
                               uint64_t seed,
                               const GPUSystem& system,
                               ThreadPool& tp)
    : mainThreadPool(tp)
    , gpuSystem(system)
    , generatorCount(rip.regionMax - rip.regionMin)
    , currentRange{Vector2ui::Zero(), Vector2ui::Zero()}
    , hostMem(gpuSystem)
    , deviceMem(gpuSystem.AllGPUs(), 2_MiB, 32_MiB)
{
    // Use seed to generate seeds;
    size_t gpuRNGCountMax = maxGPUPresentRNGCount.Multiply();
    MemAlloc::AllocateMultiData(Tie(dBackupStatesLocal, dMainStatesLocal),
                                deviceMem,
                                {gpuRNGCountMax , gpuRNGCountMax});

    size_t totalRNGCount = generatorCount.Multiply();
    MemAlloc::AllocateMultiData(Tie(hBackupStatesAll, hMainStatesAll),
                                hostMem,
                                {totalRNGCount , totalRNGCount});

    // Initialize common global state
    Vector2ui res = rip.resolution;
    uint32_t resMaxPow2 = Math::NextPowerOfTwo(res[res.Maximum()]);
    globalState.resMaxBits = Bit::RequiredBitsToRepresent(resMaxPow2) - 1;
    globalState.initialMaxSPP = initialMaxSampleCount;

    // These are const to catch race conditions etc.
    // TODO: Change this later
    uint32_t seed32 = static_cast<uint32_t>((seed >> 32) ^ (seed & 0xFFFFFFFF));
    const std::mt19937 rng0(seed32);
    std::mt19937 rngTemp = rng0;

    auto hMainStates = hMainStatesAll;
    auto future0 = tp.SubmitBlocks(uint32_t(totalRNGCount),
    [&rng0, hMainStates, rip](uint32_t start, uint32_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng0;
        rngLocal.discard(start);
        for(uint32_t i = start; i < end; i++)
        {
            //uint32_t xi = rngLocal();
            uint32_t xi = 0;

            Vector2ui regionSize = rip.regionMax - rip.regionMin;
            Vector2ui localPixelOffset = Vector2ui(i / regionSize[0],
                                                   i % regionSize[0]);
            Vector2ui localPixelIndex = rip.regionMin + localPixelOffset;
            using Graphics::MortonCode::Compose2D;
            uint64_t code = Compose2D<uint64_t>(localPixelIndex);
            ZSobolDetail::LocalState localState =
            {
                .pixelMortonCode = code,
                .sampleIndex = std::numeric_limits<uint32_t>::max(),
                .seed = xi
            };
            hMainStates[i] = localState;
        }
    }, 4u);
    future0.WaitAll();

    // Do discarding after issue (it should be logN for PRNGS)
    rngTemp.discard(totalRNGCount);
    const std::mt19937 rng1 = rngTemp;

    auto hBackupStates = hBackupStatesAll;
    auto future1 = tp.SubmitBlocks(uint32_t(totalRNGCount),
    [&rng1, hBackupStates](uint32_t start, uint32_t end)
    {
        // Local copy to the stack
        // (same functor will be run with different threads, so this
        // prevents the race condition)
        auto rngLocal = rng1;
        rngLocal.discard(start);
        for(uint32_t i = start; i < end; i++)
        {
            auto xi = rngLocal();
            hBackupStates[i] = BackupRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future1.WaitAll();


    ////
    //std::vector<Vector2> points;

    //Vector2ui reso = Vector2ui(32, 32);
    //uint32_t resMax = reso[res.Maximum()];
    //uint32_t sppMax = 32;

    //ZSobolDetail::GlobalState gs
    //{
    //    .initialMaxSPP = sppMax,
    //    .resMaxBits = Bit::RequiredBitsToRepresent(resMax) - 1
    //};

    //for(uint32_t y = 0; y < 2; ++y)
    //for(uint32_t x = 0; x < 2; ++x)
    //for(uint32_t s = 0; s < sppMax; ++s)
    //{
    //        using Graphics::MortonCode::Compose2D;
    //        ZSobolDetail::LocalState ls
    //        {
    //            .pixelMortonCode = Compose2D<uint64_t>(Vector2ui(x, y)),
    //            .sampleIndex = s,
    //            .seed = 0
    //        };
    //        ZSobolDetail::ZSobol sampler(ls, gs);

    //        uint32_t u0 = sampler.Next(0);
    //        uint32_t u1 = sampler.Next(1);
    //        Vector2 xi = Vector2(RNGFunctions::ToFloat01<Float>(u0),
    //                             RNGFunctions::ToFloat01<Float>(u1));
    //        points.push_back(xi);
    //}

    //uint32_t i = 0;
    //for(const auto& p : points)
    //{
    //    printf("%f, %f\n", p[0], p[1]);
    //    i++;
    //    //
    //    if(i % sppMax == 0) printf("-----\n");
    //}

}

void RNGGroupZSobol::SetupRange(Vector2ui rangeStart,
                                Vector2ui rangeEnd,
                                const GPUQueue& queue)
{
        // Skip if exactly the same range is requested
    if(rangeStart == currentRange[0] &&
       rangeEnd == currentRange[1]) return;

    auto GenMemcpy2DParams = [&, this](const std::array<Vector2ui, 2>& rangeIn)
    {
        size_t gpuStride = (rangeIn[1] - rangeIn[0])[0];
        size_t hostStride = generatorCount[0];
        //
        size_t gpuOffset = 0;
        size_t hostOffset = rangeIn[0][1] * hostStride + rangeIn[0][0];
        return std::array{gpuStride, gpuOffset, hostStride, hostOffset};
    };

    // Copy churned rngs back to the host
    if(currentRange[0] != Vector2ui::Zero() ||
       currentRange[1] != Vector2ui::Zero())
    {
        auto [gpuStride, gpuOffset,
              hostStride, hostOffset] = GenMemcpy2DParams(currentRange);

        queue.MemcpyAsync2D(hBackupStatesAll.subspan(hostOffset), hostStride,
                            ToConstSpan(dBackupStatesLocal), gpuStride,
                            (currentRange[1] - currentRange[0]));
        queue.MemcpyAsync2D(hMainStatesAll.subspan(hostOffset), hostStride,
                            ToConstSpan(dMainStatesLocal), gpuStride,
                            (currentRange[1] - currentRange[0]));
    }
    //
    {
        currentRange = {rangeStart, rangeEnd};
        auto [gpuStride, gpuOffset,
              hostStride, hostOffset] = GenMemcpy2DParams(currentRange);

        queue.MemcpyAsync2D(dBackupStatesLocal, gpuStride,
                            ToConstSpan(hBackupStatesAll.subspan(hostOffset)),
                            hostStride, (currentRange[1] - currentRange[0]));
        queue.MemcpyAsync2D(dMainStatesLocal, gpuStride,
                            ToConstSpan(hMainStatesAll.subspan(hostOffset)),
                            hostStride, (currentRange[1] - currentRange[0]));
    }
}

void RNGGroupZSobol::GenerateNumbers(// Output
                                     Span<RandomNumber> dNumbersOut,
                                     // Constants
                                     Vector2ui dimensionRange,
                                     const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersZSobol>
    (
        "KCGenRandomNumbersZSobol"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dimensionRange,
        globalState
    );
}

void RNGGroupZSobol::GenerateNumbersIndirect(// Output
                                             Span<RandomNumber> dNumbersOut,
                                             // Input
                                             Span<const RayIndex> dIndices,
                                             // Constants
                                             Vector2ui dimensionRange,
                                             const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    uint32_t usedGenCount = static_cast<uint32_t>(dIndices.size());
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersZSobolIndirect>
    (
        "KCGenRandomNumbersZSobolIndirect"sv,
        DeviceWorkIssueParams{.workCount = usedGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dIndices,
        dimensionRange,
        globalState
    );
}

void RNGGroupZSobol::GenerateNumbersIndirect(// Output
                                             Span<RandomNumber> dNumbersOut,
                                             // Input
                                             Span<const RayIndex> dIndices,
                                             Span<const uint32_t> dDimensionStart,
                                             // Constants
                                             uint32_t dimensionCount,
                                             const GPUQueue& queue) const
{
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersZSobolIndirectDynamicDim>
        (
            "KCGenRandomNumbersPCG32Indirect"sv,
            DeviceWorkIssueParams{.workCount = localGenCount},
            //
            dNumbersOut,
            dMainStatesLocal.subspan(0, localGenCount),
            dIndices,
            dDimensionStart,
            dimensionCount,
            globalState
        );
}

void RNGGroupZSobol::IncrementSampleId(const GPUQueue& queue) const
{
    DeviceAlgorithms::InPlaceTransform
    (
        dMainStatesLocal,
        queue,
        [] MRAY_GPU(ZSobolDetail::LocalState& ls) -> void
        {
            ls.sampleIndex++;
        }
    );
}

void RNGGroupZSobol::IncrementSampleIdIndirect(Span<const RayIndex> dIndices,
                                               const GPUQueue& queue) const
{
    DeviceAlgorithms::InPlaceTransformIndirect
    (
        dMainStatesLocal,
        dIndices,
        queue,
        [] MRAY_GPU(ZSobolDetail::LocalState& ls) -> void
        {
            ls.sampleIndex++;
        }
    );
}

Span<BackupRNGState>
RNGGroupZSobol::GetBackupStates()
{
    assert(currentRange[1] != Vector2ui::Zero());
    size_t localRNGCount = (currentRange[1] - currentRange[0]).Multiply();
    return dBackupStatesLocal.subspan(0u, localRNGCount);
}

size_t RNGGroupZSobol::GPUMemoryUsage() const
{
    return deviceMem.Size();
}
