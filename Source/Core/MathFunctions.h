#pragma once

#include "Definitions.h"
#include "MathConstants.h"
#include "MathForward.h"

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

    // Gaussian and Related Functions
    template <std::floating_point T>
    MRAY_HYBRID T Gaussian(T x, T sigma = T(1), T mu = T(0));
    template <std::floating_point T>
    MRAY_HYBRID T InvErrFunc(T x);

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

    // Simple Divisions
    template <std::integral T>
    MRAY_HYBRID constexpr T     DivideUp(T value, T alignment);
    template <uint32_t N, std::integral T>
    MRAY_HYBRID
    constexpr Vector<N, T>      DivideUp(Vector<N, T> value, Vector<N, T> divisor);

    template <std::integral T>
    MRAY_HYBRID constexpr T     NextMultiple(T value, T alignment);
    template <uint32_t N, std::integral T>
    MRAY_HYBRID
    constexpr Vector<N, T>      NextMultiple(Vector<N, T> value, Vector<N, T> divisor);

    // isinf, isnan are not constexpr
    // and CUDA does not swap to its own functions like other math
    // functions (sin, cos whatnot) So implementing
    template <std::floating_point T>
    MRAY_HYBRID bool    IsInf(T);
    template <std::floating_point T>
    MRAY_HYBRID bool    IsNan(T);

    // Misc
    template <std::integral T>
    MRAY_HYBRID constexpr T     NextPowerOfTwo(T value);
    template <std::integral T>
    MRAY_HYBRID constexpr T     PrevPowerOfTwo(T value);
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
T MathFunctions::Gaussian(T x, T sigma, T mu)
{
    assert(sigma > 0);
    using namespace MathConstants;
    static constexpr T InvSqrt2Pi = (T(1) / Sqrt2<T>()) *  (T(1) / SqrtPi<T>());
    T sigmaInv = T(1) / sigma;
    T result = InvSqrt2Pi * sigmaInv;
    T pow = (x - mu) * sigmaInv;
    result *= std::exp(T(-0.5) * pow * pow);
    return result;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::InvErrFunc(T x)
{
    // Let CUDA handle inv error function
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        if constexpr(std::is_same_v<Float, float>)
            return erfinvf(x);
        else
            return erfinv(x);
    #else
        // Checked the pbrt-v4, it has similar impl
        // of this (From a stackoverflow post).
        // https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
        //
        // https://people.maths.ox.ac.uk/gilesm/codes/erfinv/gems.pdf
        // I've checked other sites and find this
        // http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function
        // However could not implement it properly (numerical precision errors)
        // Using the stackoverflow one
        //
        Float t = std::fma(x, Float(0.0) - x, Float(1.0));
        t = std::log(t);
        Float p;
        if(std::abs(t) > Float(6.125))
        {
            p = Float(3.03697567e-10);
            p = std::fma(p, t, Float(2.93243101e-8));
            p = std::fma(p, t, Float(1.22150334e-6));
            p = std::fma(p, t, Float(2.84108955e-5));
            p = std::fma(p, t, Float(3.93552968e-4));
            p = std::fma(p, t, Float(3.02698812e-3));
            p = std::fma(p, t, Float(4.83185798e-3));
            p = std::fma(p, t, Float(-2.64646143e-1));
            p = std::fma(p, t, Float(8.40016484e-1));
        }
        else
        {
            p = Float(5.43877832e-9);
            p = std::fma(p, t, Float(1.43285448e-7));
            p = std::fma(p, t, Float(1.22774793e-6));
            p = std::fma(p, t, Float(1.12963626e-7));
            p = std::fma(p, t, Float(-5.61530760e-5));
            p = std::fma(p, t, Float(-1.47697632e-4));
            p = std::fma(p, t, Float(2.31468678e-3));
            p = std::fma(p, t, Float(1.15392581e-2));
            p = std::fma(p, t, Float(-2.32015476e-1));
            p = std::fma(p, t, Float(8.86226892e-1));
        }
        Float r = x * p;
        return r;
    #endif
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
T MathFunctions::SqrtMax(T a)
{
    return std::sqrt(std::max(T{0}, a));
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr std::array<T, 2> MathFunctions::SinCos(T v)
{
    std::array<T, 2> result;
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA

        if constexpr(std::is_same_v<T, float>)
            sincosf(v, result.data() + 0, result.data() + 1);
        else
            sincos (v, result.data() + 0, result.data() + 1);
    #else

        result[0] = std::sin(v);
        result[1] = SqrtMax(T(1) - result[0] * result[0]);

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

template <uint32_t N, std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> MathFunctions::DivideUp(Vector<N, T> value, Vector<N, T> divisor)
{
    // This do not work
    //assert(divisor != Vector<N, T>::Zero());
    // But this works? (MSVC, or CUDA?)
    // probably need "template" token somewhere
    using V = Vector<N, T>;
    assert(divisor != V::Zero());
    return (value + divisor - 1) / divisor;
}

template <std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::NextMultiple(T value, T divisor)
{
    return DivideUp(value, divisor) * divisor;
}

template <uint32_t N, std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> MathFunctions::NextMultiple(Vector<N, T> value, Vector<N, T> divisor)
{
    return DivideUp(value, divisor) * divisor;
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
bool MathFunctions::IsInf(T f)
{
    #ifndef MRAY_DEVICE_CODE_PATH_CUDA
        using namespace std;
    #endif

    return isinf(f);
}

template <std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
bool MathFunctions::IsNan(T f)
{
    #ifndef MRAY_DEVICE_CODE_PATH_CUDA
        using namespace std;
    #endif
    return isnan(f);
}

template <std::integral T>
MRAY_HYBRID constexpr T MathFunctions::NextPowerOfTwo(T value)
{
    return std::bit_ceil<T>(value);
}

template <std::integral T>
MRAY_HYBRID constexpr T MathFunctions::PrevPowerOfTwo(T value)
{
    return std::bit_floor<T>(value);
}
