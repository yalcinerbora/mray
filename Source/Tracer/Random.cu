#include "Random.h"

#include <random>

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"

#include "Core/ThreadPool.h"
#include "Core/TracerI.h"
#include "Core/GraphicsFunctions.h"

#include "SobolMatrices.h"

namespace SobolCommon
{
    // Utility
    MR_PF_DECL uint64_t MixBits(uint64_t);
    // Scramblers
    MR_PF_DECL uint32_t ScambleOwen(uint32_t v, uint32_t seed);
    MR_PF_DECL uint32_t ScambleOwenFast(uint32_t v, uint32_t seed);
}

namespace SobolDetail
{
    using namespace SobolCommon;

    MR_PF_DECL uint32_t BitwiseMatrixMult(uint64_t a, uint32_t dim,
                                          const Span<const uint32_t>& matrixList);

    class Sobol
    {
        private:
        const GlobalState&  globalState;
        uint64_t            sobolIndex;
        uint32_t            seed;

        MR_PF_DECL
        static uint32_t RollDim(uint32_t dim);

        public:
        // Constructors & Destructor
        MR_HF_DECL              Sobol(const LocalState&, const GlobalState&);
        //
        MR_HF_DECL uint32_t     Next(uint32_t dim) const;
        MR_HF_DECL Vector2ui    Next2D(uint32_t dim) const;
        MR_HF_DECL Vector3ui    Next3D(uint32_t dim) const;
    };
}

namespace ZSobolDetail
{
    using namespace SobolCommon;
    static constexpr size_t SOBOL_MATRIX_WIDTH = 52;

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

    static constexpr std::array SOBOL_32_JOE_KUO_DIM_2 =
    {
        0x80000000u, 0xC0000000u, 0x60000000u, 0x90000000u, 0xE8000000u, 0x5C000000u,
        0x8E000000u, 0xC5000000u, 0x68800000u, 0x9CC00000u, 0xEE600000u, 0x55900000u,
        0x80680000u, 0xC09C0000u, 0x60EE0000u, 0x90550000u, 0xE8808000u, 0x5CC0C000u,
        0x8E606000u, 0xC5909000u, 0x6868E800u, 0x9C9C5C00u, 0xEEEE8E00u, 0x5555C500u,
        0x8000E880u, 0xC0005CC0u, 0x60008E60u, 0x9000C590u, 0xE8006868u, 0x5C009C9Cu,
        0x8E00EEEEu, 0xC5005555u, 0x68808000u, 0x9CC0C000u, 0xEE606000u, 0x55909000u,
        0x8068E800u, 0xC09C5C00u, 0x60EE8E00u, 0x9055C500u, 0xE880E880u, 0x5CC05CC0u,
        0x8E608E60u, 0xC590C590u, 0x68686868u, 0x9C9C9C9Cu, 0xEEEEEEEEu, 0x55555555u,
        0x80000000u, 0xC0000000u, 0x60000000u, 0x90000000u
    };
    static_assert(SOBOL_32_JOE_KUO_DIM_0.size() == SOBOL_MATRIX_WIDTH &&
                  SOBOL_32_JOE_KUO_DIM_1.size() == SOBOL_MATRIX_WIDTH &&
                  SOBOL_32_JOE_KUO_DIM_2.size() == SOBOL_MATRIX_WIDTH);

    template<std::array<uint32_t, SOBOL_MATRIX_WIDTH>>
    MR_PF_DEF uint32_t BitwiseMatrixMult(uint64_t a);

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
        //
        MR_HF_DECL uint32_t     Next(uint32_t dim) const;
        MR_HF_DECL Vector2ui    Next2D(uint32_t dim) const;
        MR_HF_DECL Vector3ui    Next3D(uint32_t dim) const;
    };
}

MR_PF_DEF
uint32_t SobolCommon::ScambleOwen(uint32_t v, uint32_t seed)
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
uint32_t SobolCommon::ScambleOwenFast(uint32_t v, uint32_t seed)
{
    v = Bit::BitReverse(v);
    v ^= v * 0x3A20ADEAu;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526C56u;
    v ^= v * 0x53A22864u;
    return v;
}

// More or less 1-1 of the PBRT
// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/hash.h#L70
// Which is from this blog post.
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
MR_PF_DEF
uint64_t SobolCommon::MixBits(uint64_t v)
{
    v ^= (v >> 31ull);
    v *= 0x7FB5D329728EA185ull;
    v ^= (v >> 27ull);
    v *= 0x81DADEF4BC2DD44Dull;
    v ^= (v >> 33ull);
    return v;
};

MR_PF_DEF
uint32_t SobolDetail::BitwiseMatrixMult(uint64_t a, uint32_t dim,
                                        const Span<const uint32_t>& matrixList)
{
    auto sobolMatrix = matrixList.subspan(dim * SOBOL_MATRIX_WIDTH, SOBOL_MATRIX_WIDTH);

    uint32_t result = 0;
    for(uint32_t i = 0; i < SOBOL_MATRIX_WIDTH; i++)
    {
        if(a == 0x0) return result;
        if(a &  0x1) result ^= sobolMatrix[i];

        a >>= 1;
    }
    return result;
}

MR_PF_DEF
uint32_t SobolDetail::Sobol::RollDim(uint32_t dim)
{
    constexpr auto D = SOBOL_DIM_COUNT;

    if(dim >= D) dim -= SOBOL_DIM_COUNT;
    if(dim >= D) dim -= SOBOL_DIM_COUNT;
    else while(dim >= SOBOL_DIM_COUNT)
        dim -= SOBOL_DIM_COUNT;

    return dim;
}

MR_HF_DEF
SobolDetail::Sobol::Sobol(const LocalState& ls, const GlobalState& gs)
    : globalState(gs)
    , seed(ls.seed)
{
    // Rely on scrambling instead of sequence advancing
    sobolIndex = ls.sampleIndex;
}

MR_HF_DEF
uint32_t SobolDetail::Sobol::Next(uint32_t dim) const
{
    dim = RollDim(dim);
    uint32_t s = BitwiseMatrixMult(sobolIndex, dim, globalState.dSobolMatices);

    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    s = ScambleOwenFast(s, uint32_t(sampleHash));
    return s;
}

MR_HF_DEF
Vector2ui SobolDetail::Sobol::Next2D(uint32_t dim) const
{
    dim = RollDim(dim);
    uint32_t s0 = BitwiseMatrixMult(sobolIndex, dim + 0, globalState.dSobolMatices);
    uint32_t s1 = BitwiseMatrixMult(sobolIndex, dim + 1, globalState.dSobolMatices);

    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    s0 = ScambleOwenFast(s0, uint32_t(sampleHash & 0xFFFFFFFF));
    s1 = ScambleOwenFast(s1, uint32_t(sampleHash >> 32));
    return Vector2ui(s0, s1);
}

MR_HF_DEF
Vector3ui SobolDetail::Sobol::Next3D(uint32_t dim) const
{
    dim = RollDim(dim);
    uint32_t s0 = BitwiseMatrixMult(sobolIndex, dim + 0, globalState.dSobolMatices);
    uint32_t s1 = BitwiseMatrixMult(sobolIndex, dim + 1, globalState.dSobolMatices);
    uint32_t s2 = BitwiseMatrixMult(sobolIndex, dim + 2, globalState.dSobolMatices);

    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    static constexpr uint64_t MASK = (1 << 21) - 1;
    s0 = ScambleOwenFast(s0, uint32_t((sampleHash >>  0) & MASK));
    s1 = ScambleOwenFast(s1, uint32_t((sampleHash >> 21) & MASK));
    s2 = ScambleOwenFast(s2, uint32_t((sampleHash >> 42)       ));
    return Vector3ui(s0, s1, s2);
}

template<std::array<uint32_t, ZSobolDetail::SOBOL_MATRIX_WIDTH> SobolArray>
MR_PF_DEF
uint32_t ZSobolDetail::BitwiseMatrixMult(uint64_t a)
{
    // Let compiler decide to embed the values as immediates
    uint32_t result = 0;
    for(uint32_t i = 0; i < SOBOL_MATRIX_WIDTH; i++)
    {
        if(a == 0x0) return result;
        if(a &  0x1) result ^= SobolArray[i];

        a >>= 1;
    }
    return result;
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


        // After multiple implementations, this seems to be the
        // fastest (althought slightly)
        static constexpr std::array<uint64_t, 4> TABLE =
        {
            COMPOSE_48x2(uint64_t(0ull), 0ull, 0ull, 0ull, 0ull, 0ull,
                         1ull, 1ull, 1ull, 1ull, 1ull, 1ull,
                         2ull, 2ull, 2ull, 2ull, 2ull, 2ull,
                         3ull, 3ull, 3ull, 3ull, 3ull, 3ull),
            //
            COMPOSE_48x2(uint64_t(1ull), 1ull, 2ull, 2ull, 3ull, 3ull,
                         0ull, 0ull, 2ull, 2ull, 3ull, 3ull,
                         1ull, 1ull, 0ull, 0ull, 3ull, 3ull,
                         1ull, 1ull, 2ull, 2ull, 0ull, 0ull),
            //
            COMPOSE_48x2(uint64_t(2ull), 3ull, 1ull, 3ull, 2ull, 1ull,
                         2ull, 3ull, 0ull, 3ull, 2ull, 0ull,
                         0ull, 3ull, 1ull, 3ull, 0ull, 1ull,
                         2ull, 0ull, 1ull, 0ull, 2ull, 1ull),
            //
            COMPOSE_48x2(uint64_t(3ull), 2ull, 3ull, 1ull, 1ull, 2ull,
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
        uint32_t p = uint32_t((MixBits(higherDigits ^ dimMixer) >> 24u) % 24u);

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
    assert(dim < 3);
    if(dim == 0)
        return BitwiseMatrixMult<SOBOL_32_JOE_KUO_DIM_0>(a);
    else if(dim == 1)
        return BitwiseMatrixMult<SOBOL_32_JOE_KUO_DIM_1>(a);
    else
        return BitwiseMatrixMult<SOBOL_32_JOE_KUO_DIM_2>(a);
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
    uint64_t sampleIndex = SampleIndex(dim);
    uint32_t sample = SampleSobol32(sampleIndex, 0);
    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    sample = ScambleOwenFast(sample, uint32_t(sampleHash));
    return sample;
}

MR_HF_DEF
Vector2ui ZSobolDetail::ZSobol::Next2D(uint32_t dim) const
{
    uint64_t sampleIndex = SampleIndex(dim);
    uint32_t sample0 = SampleSobol32(sampleIndex, 0);
    uint32_t sample1 = SampleSobol32(sampleIndex, 1);

    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    sample0 = ScambleOwenFast(sample0, uint32_t(sampleHash & 0xFFFFFFFF));
    sample1 = ScambleOwenFast(sample1, uint32_t(sampleHash >> 32));
    return Vector2ui(sample0, sample1);
}

MR_HF_DEF
Vector3ui ZSobolDetail::ZSobol::Next3D(uint32_t dim) const
{
    uint64_t sampleIndex = SampleIndex(dim);
    uint32_t sample0 = SampleSobol32(sampleIndex, 0);
    uint32_t sample1 = SampleSobol32(sampleIndex, 1);
    uint32_t sample2 = SampleSobol32(sampleIndex, 2);

    using RNGFunctions::HashPCG64::Hash;
    uint64_t sampleHash = Hash(dim, seed);
    static constexpr uint64_t MASK = (1 << 21) - 1;
    sample0 = ScambleOwenFast(sample0, uint32_t((sampleHash >>  0) & MASK));
    sample1 = ScambleOwenFast(sample1, uint32_t((sampleHash >> 21) & MASK));
    sample2 = ScambleOwenFast(sample2, uint32_t((sampleHash >> 42)       ));
    return Vector3ui(sample0, sample1, sample2);
}

template<class SobolT>
MR_HF_DEF
void GenerateRNFromList(// Output
                        const Span<RandomNumber>& dNumbers,
                        // Input
                        const SobolT& rng,
                        // Constants
                        const RNRequestList& rnRequests,
                        uint32_t rngCount,
                        uint32_t rngIndex,
                        uint32_t dimOffset)
{
    uint32_t totalRequests = rnRequests.TotalRequestCount();
    uint32_t o = 0;
    for(uint32_t reqI = 0; reqI < totalRequests; reqI++)
    {
        uint32_t dim = rnRequests.DimensionOfRequest(reqI);
        switch(dim)
        {
            // Write in strided fashion to coalesce mem
            case 1:
            {
                dNumbers[rngIndex + rngCount * o] = rng.Next(dimOffset + o);
                break;
            }
            case 2:
            {
                Vector2ui numbers = rng.Next2D(dimOffset + o);
                dNumbers[rngIndex + rngCount * (o + 0)] = numbers[0];
                dNumbers[rngIndex + rngCount * (o + 1)] = numbers[1];
                break;
            }
            case 3:
            {
                Vector3ui numbers = rng.Next3D(dimOffset + o);
                dNumbers[rngIndex + rngCount * (o + 0)] = numbers[0];
                dNumbers[rngIndex + rngCount * (o + 1)] = numbers[1];
                dNumbers[rngIndex + rngCount * (o + 2)] = numbers[2];
                break;
            }
        }
        o += dim;
    }
    assert(o == rnRequests.TotalRNCount());
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersPCG32(// Output
                             MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                             // I-O
                             MRAY_GRID_CONSTANT const Span<typename PermutedCG32::State> dStates,
                             // Constants
                             MRAY_GRID_CONSTANT const uint32_t rnCount)
{
    assert(dStates.size() * rnCount <= dNumbers.size());

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
        for(uint32_t n = 0; n < rnCount; n++)
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
                                     MRAY_GRID_CONSTANT const uint32_t rnCount)
{
    assert(dNumbers.size() == dIndices.size() * rnCount);

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
        for(uint32_t n = 0; n < rnCount; n++)
        {
            // Write in strided fashion to coalesce mem
            dNumbers[i + generatorCount * n] = rng.Next();
        }
    }
}

template<class GeneratorT, class LocalStateT, class GlobalStateT>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersGeneric(// Output
                               MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                               // I-O
                               MRAY_GRID_CONSTANT const Span<const LocalStateT> dStates,
                               // Constants
                               MRAY_GRID_CONSTANT const uint32_t dimStartOffset,
                               MRAY_GRID_CONSTANT const RNRequestList rnRequests,
                               MRAY_GRID_CONSTANT const GlobalStateT globalState)
{
    assert(dStates.size() * (rnRequests.TotalRNCount()) == dNumbers.size());

    KernelCallParams kp;
    uint32_t generatorCount = uint32_t(dStates.size());
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
    {
        GeneratorT rng(dStates[i], globalState);
        GenerateRNFromList(dNumbers, rng, rnRequests,
                           generatorCount, i, dimStartOffset);
    }
}

template<class GeneratorT, class LocalStateT, class GlobalStateT>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersGenericIndirect(// Output
                                       MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                                       // I-O
                                       MRAY_GRID_CONSTANT const Span<const LocalStateT> dStates,
                                       // Input
                                       MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                       // Constants
                                       MRAY_GRID_CONSTANT const uint32_t dimStartOffset,
                                       MRAY_GRID_CONSTANT const RNRequestList rnRequests,
                                       MRAY_GRID_CONSTANT const GlobalStateT globalState)
{
    assert(dNumbers.size() == dIndices.size() * (rnRequests.TotalRNCount()));

    KernelCallParams kp;
    uint32_t generatorCount = uint32_t(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        assert(index < dStates.size());

        GeneratorT rng(dStates[index], globalState);
        GenerateRNFromList(dNumbers, rng, rnRequests,
                           generatorCount, i, dimStartOffset);
    }
}

template<class GeneratorT, class LocalStateT, class GlobalStateT>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenRandomNumbersGenericIndirectDynamicDim(// Output
                                                 MRAY_GRID_CONSTANT const Span<RandomNumber> dNumbers,
                                                 // I-O
                                                 MRAY_GRID_CONSTANT const Span<const LocalStateT> dStates,
                                                 // Input
                                                 MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                                 MRAY_GRID_CONSTANT const Span<const uint16_t> dCurrentDimensions,
                                                 // Constants
                                                 //MRAY_GRID_CONSTANT const uint32_t dimPerGenerator,
                                                 MRAY_GRID_CONSTANT const RNRequestList rnRequests,
                                                 MRAY_GRID_CONSTANT const GlobalStateT globalState)
{
    assert(dNumbers.size() == dIndices.size() * rnRequests.TotalRNCount());
    KernelCallParams kp;
    uint32_t generatorCount = uint32_t(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < generatorCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        assert(index < dStates.size());

        uint32_t dimStartOffset = dCurrentDimensions[index];
        GeneratorT rng(dStates[index], globalState);
        GenerateRNFromList(dNumbers, rng, rnRequests,
                           generatorCount, i, dimStartOffset);
    }
}


MRAY_HOST
void RNGFunctions::GenerateNumbersFromBackup(// Output
                                             Span<RandomNumber> dRNOut,
                                             // I-O
                                             Span<BackupRNGState> dBackupRNGStates,
                                             // Constants
                                             uint32_t rnCount,
                                             const GPUQueue& q)
{
    q.IssueWorkKernel<KCGenRandomNumbersPCG32>
    (
        "KCGenRandomNumbersPCG32",
        DeviceWorkIssueParams{.workCount = uint32_t(dBackupRNGStates.size())},
        //
        dRNOut,
        dBackupRNGStates,
        rnCount
    );
}

MRAY_HOST
void RNGFunctions::GenRandomNumbersFromBackupIndirect(// Output
                                                      Span<RandomNumber> dRNOut,
                                                      // I-O
                                                      Span<BackupRNGState> dBackupRNGStates,
                                                      // Input
                                                      Span<const RayIndex> dIndices,
                                                      // Constants
                                                      uint32_t rnCount,
                                                      const GPUQueue& q)
{
    q.IssueWorkKernel<KCGenRandomNumbersPCG32Indirect>
    (
        "KCGenRandomNumbersPCG32Indirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        //
        dRNOut,
        dBackupRNGStates,
        ToConstSpan(dIndices),
        rnCount
    );
}

// ============================ //
//   Independent Implementation //
// ============================ //
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
                                          uint16_t,
                                          RNRequestList rnRequests,
                                          const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<KCGenRandomNumbersPCG32>
    (
        "KCGenRandomNumbersPCG32"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        rnRequests.TotalRNCount()
    );
}

void RNGGroupIndependent::GenerateNumbersIndirect(// Output
                                                  Span<RandomNumber> dNumbersOut,
                                                  // Input
                                                  Span<const RayIndex> dIndices,
                                                  // Constants
                                                  uint16_t,
                                                  RNRequestList rnRequests,
                                                  const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    // Independent RNG do not have a notion of dimensions (or has a single
    // dimension)
    // so we disregard range, and give single random numbers
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
        rnRequests.TotalRNCount()
    );
}

void RNGGroupIndependent::GenerateNumbersIndirect(// Output
                                                  Span<RandomNumber> dNumbersOut,
                                                  // Input
                                                  Span<const RayIndex> dIndices,
                                                  Span<const uint16_t>,
                                                  // Constants
                                                  RNRequestList rnRequests,
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
        rnRequests.TotalRNCount()
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

// ======================== //
//   Sobol Implementation   //
// ======================== //
RNGGroupSobol::RNGGroupSobol(const RenderImageParams& rip,
                             Vector2ui maxGPUPresentRNGCount,
                             uint32_t,
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
    MemAlloc::AllocateMultiData(Tie(dBackupStatesLocal, dMainStatesLocal,
                                    dSobolMatrices),
                                deviceMem,
                                {gpuRNGCountMax , gpuRNGCountMax,
                                 SobolDetail::SOBOL_DATA_SIZE});

    size_t totalRNGCount = generatorCount.Multiply();
    MemAlloc::AllocateMultiData(Tie(hBackupStatesAll, hMainStatesAll),
                                hostMem,
                                {totalRNGCount , totalRNGCount});

    const GPUQueue& trasferQueue = system.BestDevice().GetTransferQueue();
    trasferQueue.MemcpyAsync(dSobolMatrices, Span<const uint32_t >(SobolDetail::SobolMatrices));

    // Initialize common global state
    globalState.resolution = rip.resolution;
    globalState.dSobolMatices = ToConstSpan(dSobolMatrices);

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
            uint32_t xi = uint32_t(rngLocal());
            Vector2ui regionSize = rip.regionMax - rip.regionMin;
            Vector2ui localPixelOffset = Vector2ui(i % regionSize[0],
                                                   i / regionSize[0]);
            Vector2ui globalPixelIndex = rip.regionMin + localPixelOffset;
            SobolDetail::LocalState localState =
            {
                .pixel = globalPixelIndex ,
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
            auto xi = uint32_t(rngLocal());
            hBackupStates[i] = BackupRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future1.WaitAll();
}

void RNGGroupSobol::SetupRange(Vector2ui rangeStart,
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

void RNGGroupSobol::GenerateNumbers(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Constants
                                    uint16_t dimensionStart,
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    using namespace SobolDetail;
    queue.IssueWorkKernel<KCGenRandomNumbersGeneric<Sobol, LocalState, GlobalState>>
    (
        "KCGenRandomNumbersSobol"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dimensionStart,
        rnRequests,
        globalState
    );
}

void RNGGroupSobol::GenerateNumbersIndirect(// Output
                                            Span<RandomNumber> dNumbersOut,
                                            // Input
                                            Span<const RayIndex> dIndices,
                                            // Constants
                                            uint16_t dimensionStart,
                                            RNRequestList rnRequests,
                                            const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    uint32_t usedGenCount = static_cast<uint32_t>(dIndices.size());
    using namespace std::string_view_literals;
    using namespace SobolDetail;
    static constexpr auto Kernel = KCGenRandomNumbersGenericIndirect<Sobol, LocalState, GlobalState>;
    queue.IssueWorkKernel<Kernel>
    (
        "KCGenRandomNumbersSobolIndirect"sv,
        DeviceWorkIssueParams{.workCount = usedGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dIndices,
        dimensionStart,
        rnRequests,
        globalState
    );
}

void RNGGroupSobol::GenerateNumbersIndirect(// Output
                                            Span<RandomNumber> dNumbersOut,
                                            // Input
                                            Span<const RayIndex> dIndices,
                                            Span<const uint16_t> dDimensionStartOffsets,
                                            // Constants
                                            RNRequestList rnRequests,
                                            const GPUQueue& queue) const
{
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    static constexpr auto Kernel = KCGenRandomNumbersGenericIndirectDynamicDim
    <
        SobolDetail::Sobol,
        SobolDetail::LocalState,
        SobolDetail::GlobalState
    >;
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<Kernel>
    (
        "KCGenRandomNumbersSobolIndirect"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dIndices,
        dDimensionStartOffsets,
        rnRequests,
        globalState
    );
}

void RNGGroupSobol::IncrementSampleId(const GPUQueue& queue) const
{
    DeviceAlgorithms::InPlaceTransform
    (
        dMainStatesLocal,
        queue,
        [] MRAY_GPU(SobolDetail::LocalState& ls) -> void
        {
            ls.sampleIndex++;
        }
    );
}

void RNGGroupSobol::IncrementSampleIdIndirect(Span<const RayIndex> dIndices,
                                              const GPUQueue& queue) const
{
    DeviceAlgorithms::InPlaceTransformIndirect
    (
        dMainStatesLocal,
        dIndices,
        queue,
        [] MRAY_GPU(SobolDetail::LocalState& ls) -> void
        {
            ls.sampleIndex++;
        }
    );
}

Span<BackupRNGState>
RNGGroupSobol::GetBackupStates()
{
    assert(currentRange[1] != Vector2ui::Zero());
    size_t localRNGCount = (currentRange[1] - currentRange[0]).Multiply();
    return dBackupStatesLocal.subspan(0u, localRNGCount);
}

size_t RNGGroupSobol::GPUMemoryUsage() const
{
    return deviceMem.Size();
}

// ======================== //
//  ZSobol Implementation   //
// ======================== //
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
            uint32_t xi = uint32_t(rngLocal());
            Vector2ui regionSize = rip.regionMax - rip.regionMin;
            Vector2ui localPixelOffset = Vector2ui(i % regionSize[0],
                                                   i / regionSize[0]);
            Vector2ui globalPixelIndex = rip.regionMin + localPixelOffset;
            using Graphics::MortonCode::Compose2D;
            uint64_t code = Compose2D<uint64_t>(globalPixelIndex);
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
            auto xi = uint32_t(rngLocal());
            hBackupStates[i] = BackupRNG::GenerateState(uint32_t(xi));
        }
    }, 4u);
    future1.WaitAll();
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
                                     uint16_t dimensionStart,
                                     RNRequestList rnRequests,
                                     const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    using namespace std::string_view_literals;
    using namespace ZSobolDetail;
    queue.IssueWorkKernel<KCGenRandomNumbersGeneric<ZSobol, LocalState, GlobalState>>
    (
        "KCGenRandomNumbersZSobol"sv,
        DeviceWorkIssueParams{.workCount = localGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dimensionStart,
        rnRequests,
        globalState
    );
}

void RNGGroupZSobol::GenerateNumbersIndirect(// Output
                                             Span<RandomNumber> dNumbersOut,
                                             // Input
                                             Span<const RayIndex> dIndices,
                                             // Constants
                                             uint16_t dimensionStart,
                                             RNRequestList rnRequests,
                                             const GPUQueue& queue) const
{
    assert(currentRange[1] != Vector2ui::Zero());

    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    uint32_t usedGenCount = static_cast<uint32_t>(dIndices.size());
    using namespace std::string_view_literals;
    using namespace ZSobolDetail;
    static constexpr auto Kernel = KCGenRandomNumbersGenericIndirect<ZSobol, LocalState, GlobalState>;
    queue.IssueWorkKernel<Kernel>
    (
        "KCGenRandomNumbersZSobolIndirect"sv,
        DeviceWorkIssueParams{.workCount = usedGenCount},
        //
        dNumbersOut,
        dMainStatesLocal.subspan(0, localGenCount),
        dIndices,
        dimensionStart,
        rnRequests,
        globalState
    );
}

void RNGGroupZSobol::GenerateNumbersIndirect(// Output
                                             Span<RandomNumber> dNumbersOut,
                                             // Input
                                             Span<const RayIndex> dIndices,
                                             Span<const uint16_t> dDimensionStartOffsets,
                                             // Constants
                                             RNRequestList rnRequests,
                                             const GPUQueue& queue) const
{
    uint32_t localGenCount = (currentRange[1] - currentRange[0]).Multiply();
    static constexpr auto Kernel = KCGenRandomNumbersGenericIndirectDynamicDim
    <
        ZSobolDetail::ZSobol,
        ZSobolDetail::LocalState,
        ZSobolDetail::GlobalState
    >;
    using namespace std::string_view_literals;
    queue.IssueWorkKernel<Kernel>
        (
            "KCGenRandomNumbersZSobolIndirect"sv,
            DeviceWorkIssueParams{.workCount = localGenCount},
            //
            dNumbersOut,
            dMainStatesLocal.subspan(0, localGenCount),
            dIndices,
            dDimensionStartOffsets,
            rnRequests,
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
