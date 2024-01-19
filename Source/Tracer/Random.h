#pragma once

#include <cstdint>
#include "Core/Vector.h"
#include "Core/BitFunctions.h"
#include "Device/GPUSystem.h"

namespace RNGFunctions
{
    template<std::floating_point F, std::unsigned_integral T>
    static F ToFloat01(T);

    // TODO: Implement double as well
    template<>
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

// Xoroshiro64
// https://prng.di.unimi.it/xoroshiro64starstar.c
// This RNG predominantly be used as a "backup" generator
// Each ray probably has one of these when it requires an undeterministic
// amount of random numbers.
//
// This occurs over the alpha testing over the scene.
class Xoroshiro64
{
    public:
    using State = Vector2ui;

    private:
    State&                  dState;
    State                   rState;

    public:
    MRAY_HYBRID             Xoroshiro64(State& dState);
    MRAY_HYBRID             ~Xoroshiro64();

    MRAY_HYBRID uint32_t    Next();
    MRAY_HYBRID Float       NextFloat();
};

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
Xoroshiro64::Xoroshiro64(State& dState)
    : dState(dState)
    , rState(rState)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
Xoroshiro64::~Xoroshiro64()
{
    dState = rState;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t Xoroshiro64::Next()
{
    using namespace BitFunctions;

    uint32_t result = RotateLeft(rState[0] * 0x9E3779BB, 5u);
    result *= 5u;

    rState[0] = RotateLeft(rState[0], 26u);
    rState[0] ^= rState[1];
    rState[0] ^= rState[1] << 9;
    rState[1] = RotateLeft(rState[1], 13u);

    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Xoroshiro64::NextFloat()
{
    uint32_t nextInt = Next();
    return RNGFunctions::ToFloat01<Float>(nextInt);
}

// TODO: Make this compile-time constant
using BackupRNG = Xoroshiro64;
using BackupRNGState = typename Xoroshiro64::State;

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