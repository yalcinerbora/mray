#pragma once

#include "Definitions.h"
#include "MathConstants.h"

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
    template <std::integral T>
    MRAY_HYBRID constexpr T     NextMultiple(T value, T alignment);

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
    static constexpr T InvSqrtPi = T(1) / MathConstants::SqrtPi<Float>();
    T sigmaInv = T(1) / sigma;
    T result = InvSqrtPi * sigmaInv;
    T pow = (x - mu) * sigmaInv;
    result *= std::expf(T(-0.5) * pow * pow);
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
        // https://people.maths.ox.ac.uk/gilesm/codes/erfinv/gems.pdf
        // I've checked other sites and find this
        // http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function
        // since this is on CPU, let's do Newton-Raphson refinement

        using V4 = std::array<T, 4>;
        using V2 = std::array<T, 4>;
        T result;
        if(std::abs(x) <= T(0.7))
        {
            static constexpr
            V4 a = {Float(0.886226899), Float(-1.645349621),
                    Float(0.914624893), Float(-0.140543331)};
            static constexpr
            V4 b = {Float(-2.118377725), Float(1.442710462),
                    Float(-0.329097515), Float(0.012229801)};

            Float xSqr = x * x;
            result = a[3] * xSqr + a[2];
            result = result * xSqr + a[1];
            result = result * xSqr + a[0];
            Float denom = b[3] * xSqr + b[2];
            denom = denom * xSqr + b[1];
            denom = denom * xSqr + b[0];
            denom = denom * xSqr + Float(1);
            result /= denom;
        }
        else
        {
            static constexpr
            V4 c = {Float(-1.970840454), Float(-1.62490649),
                    Float(3.429567803), Float(1.641345311)};
            static constexpr
            V2 d = {Float(3.543889200), Float(1.637067800)};

            Float z = std::abs(x);
            Float y = std::sqrt((-std::log(1 - z) * Float(0.5)));
            result = c[3]   * y + c[2];
            result = result * y + c[1];
            result = result * y + c[0];

            // Do not continue if estimate is already inf
            if(MathFunctions::IsInf(result)) return result;

            Float denom = d[1] * y + d[0];
            denom = denom * y + Float(1);
            result /= denom;
        }
        result = std::copysign(result, x);

        static constexpr T C = T(2) / MathConstants::SqrtPi<T>();
        result -= (std::erf(result) - x) / (C * std::exp(-result * result));
        result -= (std::erf(result) - x) / (C * std::exp(-result * result));
        return result;
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

template <std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T MathFunctions::NextMultiple(T value, T divisor)
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
