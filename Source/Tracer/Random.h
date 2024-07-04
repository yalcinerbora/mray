#pragma once

#include <cstdint>
#include "Core/Vector.h"
#include "Core/BitFunctions.h"
#include "Device/GPUSystem.h"

namespace RNGFunctions::HashPCG64
{
    // I've checked the PBRT, and decided to use Murmur first
    // but arbitrary byte size made the NVCC not happy
    // (no register-space code generation, unaligned loads etc)
    // Insted I've come up with this paper (JCGT has nice gems)
    // https://jcgt.org/published/0009/03/02/
    //
    // PGC looks like a good choice, so we will implement it as a
    // hashing function (instead of RNG)
    //
    // To make it applicable to arbitrary sizes we will recursively
    // call it as discussed in the paper (i.e., H(x_n + H(x_{n-1} + H(...)));
    //
    // This should give the compiler enough information to generate register
    // space code when data is again in register space.

    // Some type traits to dynamically bit_cast to underlying int type
    // bit_cast will handle all the quirks (hopefully)
    // TODO: I may have written this somewhere else
    // but could not find it.
    namespace Detail
    {
        template<class T>
        struct uintX
        {
            static_assert(sizeof(T) <= sizeof(uint64_t),
                          "Hashing requires the data types to be at most 64-bit!");
        };

        template<class T>
        requires(sizeof(T) == sizeof(uint8_t))
        struct uintX<T> { using type = uint8_t; };

        template<class T>
        requires(sizeof(T) == sizeof(uint16_t))
        struct uintX<T> { using type = uint16_t; };

        template<class T>
        requires(sizeof(T) == sizeof(uint32_t))
        struct uintX<T> { using type = uint32_t; };

        template<class T>
        requires(sizeof(T) == sizeof(uint64_t))
        struct uintX<T> { using type = uint64_t; };

        template<class T>
        using uintX_t = typename uintX<T>::type;

        template<class T>
        inline constexpr bool IntCastable_v = (sizeof(T) == sizeof(uint8_t)  ||
                                               sizeof(T) == sizeof(uint16_t) ||
                                               sizeof(T) == sizeof(uint32_t) ||
                                               sizeof(T) == sizeof(uint64_t));
    }

    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr uint64_t Hash(uint64_t v);

    template<class... Args>
    requires(Detail::IntCastable_v<Args> && ...)
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr uint64_t Hash(Args&&...);
}

namespace RNGFunctions
{
    template<std::floating_point F, std::unsigned_integral T>
    MRAY_HYBRID static F ToFloat01(T);

    // TODO: Implement double as well
    template<>
    MRAY_HYBRID MRAY_CGPU_INLINE
    float ToFloat01<float, uint32_t>(uint32_t v)
    {
        // Sanity Checks
        static_assert(std::numeric_limits<float>::is_iec559, "Non-standard floating point!");
        static_assert(sizeof(uint32_t) == sizeof(float), "float is not 32-bit!");

        // This is simpler version also its 24-bit instead of 23-bit
        //https://marc-b-reynolds.github.io/distribution/2017/01/17/DenseFloat.html
        float rngFloat = static_cast<float>(v >> 8);
        return  rngFloat * 0x1p-24f;
    }



    //MRAY_HYBRID
    //constexpr uint64_t MurmurHash2_32Bit(const Span<const Byte>& data, uint64_t seed)
    //{
    //    auto GetBlock = [](Span<const Byte, sizeof(uint32_t)> bytes) -> uint32_t
    //    {
    //        // Require an alignment, we will load 32-bit chunks (GPU will start
    //        // shuffling the data assuming it is not aligned)
    //        const Byte* data = std::assume_aligned<sizeof(uint32_t)>(bytes.data());
    //        // We cant memcpy or reinterpret_cast due to constexpr
    //        // Do an upcast (static_cast), then twiddle the bits.
    //        // Let compiler to figure it out.
    //        auto SC = [](Byte i) -> uint32_t {return static_cast<uint32_t>(i); };
    //        return (SC(data[0]) << CHAR_BIT * 0u | SC(data[1]) << CHAR_BIT * 1u |
    //                SC(data[2]) << CHAR_BIT * 2u | SC(data[3]) << CHAR_BIT * 3u);
    //    };

    //    constexpr uint32_t SZ_32 = sizeof(uint32_t);
    //    constexpr uint32_t SZ_64 = sizeof(uint64_t);
    //    constexpr uint32_t R = 24;
    //    constexpr uint32_t M = 0x5bd1e995;

    //    uint32_t length = static_cast<uint32_t>(data.size_bytes());
    //    uint32_t h1 = uint32_t(seed) ^ length;
    //    uint32_t h2 = uint32_t(seed >> SZ_32 * CHAR_BIT);

    //    uint32_t iterations = length / sizeof(uint64_t);
    //    for(uint32_t i = 0; i < iterations; i++)
    //    {
    //        Span<const Byte> sub = (data.subspan(i * SZ_64, SZ_64));
    //        uint32_t k1 = GetBlock(sub.subspan<0, SZ_32>());
    //        k1 *= M; k1 ^= k1 >> R; k1 *= M;
    //        h1 *= M; h1 ^= k1;

    //        uint32_t k2 = GetBlock(sub.subspan<SZ_32, SZ_32>());
    //        k2 *= M; k2 ^= k2 >> R; k2 *= M;
    //        h2 *= M; h2 ^= k2;
    //    }

    //    if((length & 0b111) >= 4)
    //    {
    //        Span<const Byte> sub = (data.subspan(iterations * sizeof(uint64_t), SZ_32));
    //        uint32_t k1 = GetBlock(sub.subspan<0, SZ_32>());
    //        k1 *= M; k1 ^= k1 >> R; k1 *= M;
    //        h1 *= M; h1 ^= k1;
    //    }

    //    uint32_t leftBytes = length & 0b11;
    //    auto finalRange = data.subspan(iterations * sizeof(uint64_t) + SZ_32, leftBytes);
    //    for(uint32_t i = 0; i < leftBytes; i++)
    //        h2 ^= static_cast<uint32_t>(finalRange[i]) << (i * CHAR_BIT);
    //    if(leftBytes != 0) h2 *= M;

    //    h1 ^= h2 >> 18; h1 *= M;
    //    h2 ^= h1 >> 22; h2 *= M;
    //    h1 ^= h2 >> 17; h1 *= M;
    //    h2 ^= h1 >> 19; h2 *= M;

    //    uint64_t h = h1;
    //    return (h << 32) | h2;
    //}

    //MRAY_HYBRID
    //template<class... Args>
    //constexpr uint64_t MurmurHash2(Args&&... args)
    //{
    //    constexpr uint32_t TOTAL = (sizeof(Args) + ...);
    //    alignas(sizeof(uint32_t)) std::array<Buffer, TOTAL> Buffer;

    //    size_t offset = 0;
    //    ((std::memcpy(Buffer.data() + offset), &args),
    //     (void)(offset += sizeof(args))), ...);

    //    return MurmurHas2_64Bit(Buffer, 0);

    //}
}

template <std::unsigned_integral T>
struct RNGDispenserT
{
    private:
    Span<const T>       randomNumbers;
    uint32_t            globalId;
    uint32_t            stride;

    public:
    MRAY_HYBRID         RNGDispenserT(Span<const T> numbers,
                                      uint32_t globalId,
                                      uint32_t stride);

    template<uint32_t I>
    MRAY_HYBRID Float   NextFloat();
    template<uint32_t I>
    MRAY_HYBRID Vector2 NextFloat2D();
};

using RNGDispenser = RNGDispenserT<uint32_t>;

// PCG with 32bit state
// https://www.pcg-random.org
// This RNG predominantly be used as a "backup" generator,
// It has minimal state and andvancement capabilities
// (which will be usefull on volume rendering I hope)
//
// Each ray probably has one of these when it requires an undeterministic
// amount of random numbers. (Anyhit shader (alpha blend), volume rendering)
// Minimal state is usefull since we will pass these as ray payload on optix
// 64-bit version may be used here but we will try to minimize register
// usage for this generator. Because we already have high quality sample
// generator from Sobol etc.
//
// This class is designed for register space usage,
// upon destruction it will save its state to the device memory that is given.
// So there is no solid class throughout the execution of the program.
// Use static function "GenState" to create the seed.
class PermutedCG32
{
    static constexpr uint32_t Multiplier = 747796405u;
    static constexpr uint32_t Increment = 2891336453u;

    public:
    using State = uint32_t;

    MRAY_HYBRID
    static State    GenerateState(uint32_t seed);

    private:
    State&          dState;
    State           rState;

    MRAY_HYBRID
    static State    Step(State r);
    MRAY_HYBRID
    static uint32_t Permute(State r);

    public:
    MRAY_HYBRID     PermutedCG32(State& dState);
    MRAY_HYBRID     ~PermutedCG32();

    MRAY_HYBRID
    uint32_t        Next();

    MRAY_HYBRID
    Float           NextFloat();

    MRAY_HYBRID
    void            Advance(uint32_t delta);
};

namespace RNGFunctions
{

// 64->64 variant used to hash 64 bit types
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint64_t HashPCG64::Hash(uint64_t v)
{
    constexpr uint64_t Multiplier = 6364136223846793005ull;
    constexpr uint64_t Increment = 1442695040888963407ull;
    uint64_t s = v * Multiplier + Increment;
    uint64_t word = ((s >> ((s >> 59u) + 5u)) ^ s);
    word *= 12605985483714917081ull;
    return (word >> 43u) ^ word;
}

template<class... Args>
requires(HashPCG64::Detail::IntCastable_v<Args> && ...)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint64_t HashPCG64::Hash(Args&&... args)
{
    using namespace HashPCG64::Detail;
    // According to the paper recursive hash and add
    // is recommended.
    // Abusing comma operator expansion here,
    // I've learned this recently and wanted to use somewhere
    // This was somewhat a good place to use this.
    // This may improve compilation times when sizeof...(Args) is large?
    uint64_t v = 0;
    (
        static_cast<void>(v = Hash(std::bit_cast<uintX_t<Args>>(args) + v)),
        ...
    );
    return v;
}

}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
RNGDispenserT<T>::RNGDispenserT(Span<const T> numbers,
                                uint32_t globalId,
                                uint32_t stride)
    : randomNumbers(numbers)
    , globalId(globalId)
    , stride(stride)
{}

template <std::unsigned_integral T>
template<uint32_t I>
MRAY_HYBRID MRAY_CGPU_INLINE
Float RNGDispenserT<T>::NextFloat()
{
    assert((globalId + I * stride) < randomNumbers.size());
    uint32_t xi0 = randomNumbers[globalId + I * stride];

    return RNGFunctions::ToFloat01<Float>(xi0);
}

template <std::unsigned_integral T>
template<uint32_t I>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 RNGDispenserT<T>::NextFloat2D()
{
    assert((globalId + (I + 1) * stride) < randomNumbers.size());
    uint32_t xi0 = randomNumbers[globalId + I       * stride];
    uint32_t xi1 = randomNumbers[globalId + (I + 1) * stride];

    return Vector2(RNGFunctions::ToFloat01<Float>(xi0),
                   RNGFunctions::ToFloat01<Float>(xi1));
}

MRAY_HYBRID MRAY_CGPU_INLINE
typename PermutedCG32::State
PermutedCG32::GenerateState(uint32_t seed)
{
    State r;
    r = Step(0);
    r += seed;
    r = Step(r);
    return r;
}

MRAY_HYBRID MRAY_CGPU_INLINE
typename PermutedCG32::State
PermutedCG32::Step(State r)
{
    return r * Multiplier + Increment;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PermutedCG32::Permute(State s)
{
    uint32_t result = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (result >> 22u) ^ result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
PermutedCG32::PermutedCG32(State& dState)
    : dState(dState)
    , rState(dState)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
PermutedCG32::~PermutedCG32()
{
    dState = rState;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PermutedCG32::Next()
{
    State newState = Step(rState);
    uint32_t r = Permute(rState);
    rState = newState;
    return r;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float PermutedCG32::NextFloat()
{
    uint32_t nextInt = Next();
    return RNGFunctions::ToFloat01<Float>(nextInt);
}

MRAY_HYBRID MRAY_CGPU_INLINE
void PermutedCG32::Advance(uint32_t delta)
{
    // This is from the PCG32 code,
    // which is based on
    // Brown, "Random Number Generation
    // with Arbitrary Stride", Transactions of the American Nuclear
    // Society (Nov. 1994).
    //
    // Upon inspection, this may not be useful for advancing the state on a warp
    // local fashion since it is log(N) operations
    // We may want to advance for warp size
    // log(32) = 5 iterations, then do 32 i
    uint32_t accMult = 1u;
    uint32_t accPlus = 0u;
    uint32_t curMult = Multiplier;
    uint32_t curPlus = Increment;
    while(delta > 0)
    {
        if(delta & 0b1)
        {
            accMult *= curMult;
            accPlus = accPlus * curMult + curPlus;
        }
        curPlus = (curMult + 1u) * curPlus;
        curMult *= curMult;
        delta /= 2;
    }
    rState = accMult * rState + accPlus;
}

using BackupRNG = PermutedCG32;
using BackupRNGState = typename PermutedCG32::State;

template <class MainRNGType>
class RNGeneratorGroup
{
    public:
    using MainRNG           = MainRNGType;
    using MainRNGState      = typename MainRNG::State;

    private:
    DeviceMemory            memory;
    Span<BackupRNGState>    backupStates;
    Span<MainRNGState>      mainStates;

    public:
    RNGeneratorGroup(size_t generatorCount, uint32_t seed);


    //State



    //RNGDispenser();
};