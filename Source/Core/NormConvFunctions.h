#pragma once

#include "Math.h"

namespace NormConversion
{
    template<FloatC R, std::unsigned_integral T>
    MR_PF_DECL R FromUNorm(T in) noexcept;

    template<FloatC R, uint32_t Bits,
                std::unsigned_integral T>
    MR_PF_DECL R FromUNormVaryingInsane(T value) noexcept;

    template<FloatC R, std::unsigned_integral T>
    MR_PF_DECL R FromUNormVarying(T in, T bits) noexcept;

    template<std::unsigned_integral T, FloatC R>
    MR_PF_DECL T ToUNorm(R in) noexcept;

    template<std::unsigned_integral T, FloatC R>
    MR_PF_DECL T ToUNormVarying(R in, T bits) noexcept;

    template<FloatC R, std::signed_integral T>
    MR_PF_DECL R FromSNorm(T in) noexcept;

    template<std::signed_integral T, FloatC R>
    MR_PF_DECL T ToSNorm(R in) noexcept;
}

template<FloatC R, std::unsigned_integral T>
MR_PF_DEF R NormConversion::FromUNorm(T in) noexcept
{
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R MIN = R(std::numeric_limits<T>::min());
    constexpr R DELTA = 1 / (MAX - MIN);
    // TODO: Specialize using intrinsics maybe?
    // For GPU (GPUs should have these?)
    // Also check more precise way to do this (if available?)
    R result = MIN + static_cast<R>(in) * DELTA;
    return result;
}

template<FloatC R, uint32_t BITS, std::unsigned_integral T>
MR_PF_DEF R NormConversion::FromUNormVaryingInsane(T value) noexcept
{
    // We can do divide an all, but we get our hands dirty for
    // bisecting BC compressions for color conversion
    // So lets do a cool version of unorm generation directly
    // embedding to the mantissa of the float
     // Directly embed these to mantissa
    static_assert(std::numeric_limits<Float>::is_iec559,
                  "IEEE-754 floats are required for this function");
    using IntT = std::conditional_t<(std::is_same_v<Float, float>),
                                    uint32_t,
                                    uint64_t>;
    constexpr IntT MANTISSA_SIZE = std::is_same_v<Float, float>
                                    ? 23u
                                    : 52u;
    static_assert(BITS <= MANTISSA_SIZE, "Bit count exceeds mantissa");
    assert(value >> BITS == 0);
    // Set the exponent side via the compiler
    // lets not get fancy (thanks to bit_cast)
    IntT rBits = std::bit_cast<IntT>(R(1));
    IntT input = value;
    input <<= (MANTISSA_SIZE - BITS);
    // Here is the fancy part we copy the value to the mantissa
    // starting from MSB of the mantissa
    constexpr IntT StampCount = Math::DivideUp(MANTISSA_SIZE, BITS);
    MRAY_UNROLL_LOOP
    for(IntT i = 0; i < StampCount; i++)
    {
        rBits |= input;
        input >>= BITS;
    }
    // All this work makes sense when Float is 1.0, extract 1.
    //
    // Is this worth it? Definitely no.
    // NVCC recommends to not mix float -> int conversions, we did that
    // a classical approach
    // R(input) / R((1 << bitSize) - 1) also have type conversions
    // so we did not gained anything
    // Furthermore, this is value not compatible with this approach.
    //
    // But no floating point division so its performant even with a loop?
    // probably not. So do not do this :)
    return R(1) - std::bit_cast<R>(rBits);
}

template<FloatC R, std::unsigned_integral T>
MR_PF_DEF R NormConversion::FromUNormVarying(T in,T bits) noexcept
{
    R max = R((T(1) << bits) - T(1));
    R min = R(std::numeric_limits<T>::min());
    R delta = 1 / (max - min);
    // TODO: Specialize using intrinsics maybe?
    // For GPU (GPUs should have these?)
    // Also check more precise way to do this (if available?)
    R result = min + static_cast<R>(in) * delta;
    return result;
}

template<std::unsigned_integral T, FloatC R>
MR_PF_DEF T NormConversion::ToUNorm(R in) noexcept
{
    #ifndef MRAY_DEVICE_CODE_PATH
        using std::round;
    #endif

    assert(in >= R(0) && in <= R(1));
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R MIN = R(std::numeric_limits<T>::min());
    constexpr R DIFF = (MAX - MIN);

    in *= DIFF;
    in = round(in);
    return T(in);
}

template<std::unsigned_integral T, FloatC R>
MR_PF_DEF T NormConversion::ToUNormVarying(R in, T bits) noexcept
{
    #ifndef MRAY_DEVICE_CODE_PATH
        using std::round;
    #endif

    assert(in >= R(0) && in <= R(1));
    R max = R((T(1) << bits) - T(1));
    R min = R(std::numeric_limits<T>::min());
    R diff = (max - min);

    in *= diff;
    in = round(in);
    return T(in);
}

template<FloatC R, std::signed_integral T>
MR_PF_DEF R NormConversion::FromSNorm(T in) noexcept
{
    // Math representation (assuming T is char)
    // [-128, ..., 0, ..., 127]
    // However due to 2's complement, bit to data layout is
    // [0, ..., 127, -128, ..., -1]
    // DirectX representation is
    // [0, ..., 127, -127, -127, ..., -1]
    //               ----^-----
    //               notice the two -127's
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion
    // we will use DX version since these will be used to
    // fetch BC textures.
    static_assert(std::numeric_limits<T>::max() +
                  std::numeric_limits<T>::min() == -1,
                  "Not two's complement int type?");
    constexpr T MIN = std::numeric_limits<T>::min();
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R DELTA = R(1) / (MAX - R(0));

    in = (in == MIN) ? (in + 1) : in;
    R result = R(in) * DELTA;
    return result;
}

template<std::signed_integral T, FloatC R>
MR_PF_DEF T NormConversion::ToSNorm(R in) noexcept
{
    #ifndef MRAY_DEVICE_CODE_PATH
        using std::round;
    #endif

    assert(in >= R(-1) && in <= R(1));
    // Check "FromSNorm" for more info
    //
    static_assert(std::numeric_limits<T>::max() +
                  std::numeric_limits<T>::min() == -1,
                  "Not two's complement int type?");
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R DIFF = (MAX - R(0));

    in *= DIFF;
    in = round(in);
    return T(in);
}