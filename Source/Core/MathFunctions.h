#pragma once

#include "Definitions.h"
#include <cmath>
#include <bit>

namespace MathFunctions
{
    template <std::integral T>
    MRAY_HYBRID constexpr T Clamp(T, T minVal, T maxVal);
    template <std::floating_point T>
    MRAY_HYBRID T           Clamp(T, T minVal, T maxVal);
    template <std::floating_point T>
    MRAY_HYBRID constexpr T Lerp(T a, T b, T t);
    template <std::floating_point T>
    MRAY_HYBRID T           Smoothstep(T a, T b, T t);

    // This pattern comes out to0 many times due to
    // numeric calculations. It just prevents a to be negative.
    // Basically "sqrt(max(0, a))".
    template <std::floating_point T>
    MRAY_HYBRID  T              SqrtMax(T a);


    template <std::integral T>
    MRAY_HYBRID constexpr T     DivideUp(T value, T alignment);
    template <std::integral T>
    MRAY_HYBRID constexpr T     NextMultiple(T value, T alignment);

    template <std::integral T>
    MRAY_HYBRID constexpr T     NextPowerOfTwo(T value);
}

template <std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::Clamp(T v, T minVal, T maxVal)
{
    v = (v < minVal) ? minVal : v;
    v = (v > maxVal) ? maxVal : v;
    return v;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::Clamp(T v, T minVal, T maxVal)
{
    return fmin(fmax(minVal, v), maxVal);
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::Lerp(T a, T b, T t)
{
    return a * (T{1} - t) + b * t;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::Smoothstep(T a, T b, T t)
{
    // https://en.wikipedia.org/wiki/Smoothstep
    t = Clamp((t - a) / (b - a), T{0}, T{1});
    return t * t * (T{3} - T{2} * t);
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::SqrtMax(T a)
{
    return sqrt(fmax(Float{0}, a));
}

template <std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::DivideUp(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

template <std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::NextMultiple(T value, T divisor)
{
    return DivideUp(value, divisor) * divisor;
}

template <std::integral T>
MRAY_HYBRID constexpr T MathFunctions::NextPowerOfTwo(T value)
{
    return std::bit_floor<T>(value);
}

