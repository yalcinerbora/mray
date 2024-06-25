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
    MRAY_HYBRID constexpr T Smoothstep(T a, T b, T t);

    template <std::floating_point T>
    MRAY_HYBRID T NextFloat(T a);
    template <std::floating_point T>
    MRAY_HYBRID T PrevFloat(T a);

    // TODO: Although not math, put it here
    // Similar to clamp but rolls
    template <std::signed_integral T>
    MRAY_HYBRID constexpr T Roll(T, T minVal, T maxVal);

    // This pattern comes out too many times in
    // numeric calculations. Due to precision errors,
    // sometimes mathematicall correct square roots
    // may result in NaN.
    // Basically "sqrt(max(0, a))".
    template <std::floating_point T>
    MRAY_HYBRID  T              SqrtMax(T a);

    // Trigonometry
    template <std::floating_point T>
    MRAY_HYBRID
    constexpr std::array<T, 2>  SinCos(T);

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
    assert(minVal < maxVal);
    v = (v < minVal) ? minVal : v;
    v = (v > maxVal) ? maxVal : v;
    return v;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::Clamp(T v, T minVal, T maxVal)
{
    assert(minVal < maxVal);
    return std::min(std::max(minVal, v), maxVal);
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::Lerp(T a, T b, T t)
{
    assert(t >= T(0) && t <= T(1));
    return a * (T{1} - t) + b * t;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::Smoothstep(T a, T b, T t)
{
    assert(t >= T(0) && t <= T(1));
    // https://en.wikipedia.org/wiki/Smoothstep
    t = Clamp((t - a) / (b - a), T{0}, T{1});
    return t * t * (T{3} - T{2} * t);
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::NextFloat(T a)
{
    return std::nextafter(a, std::numeric_limits<T>::max());
}
template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::PrevFloat(T a)
{
    return std::nextafter(a, -std::numeric_limits<T>::max());
}

template <std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::Roll(T v, T min, T max)
{
    assert(min < max);
    T diff = max - min;
    v -= min;
    v %= diff;
    v = (v < 0) ? diff + v : v;
    v += min;
    return v;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::SqrtMax(T a)
{
    return std::sqrt(std::max(Float{0}, a));
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr std::array<T, 2> MathFunctions::SinCos(T v)
{
    std::array<T, 2> result;
    #if defined(MRAY_GPU_BACKEND_CUDA) && defined(__CUDA_ARCH__)

        if constexpr(std::is_same_v<T, float>)
            sincosf(v, result.data() + 0, result.data() + 1);
        else
            sincos (v, result.data() + 0, result.data() + 1);
    #else

        result[0] = std::sin(v);
        result[1] = SqrtMax(Float(1) - result[0] * result[0]);

    #endif

    return result;
}

template <std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::DivideUp(T value, T divisor)
{
    assert(divisor != T(0));
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
    return std::bit_ceil<T>(value);
}

