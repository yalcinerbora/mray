#pragma once

#include "Definitions.h"
#include "MathConstants.h"
#include "MathForward.h"
#include "BitFunctions.h"

#include <numeric>
#include <cmath>
#include <cstdlib>

namespace Math
{
    template<uint32_t N>
    class MovingAverage
    {
        private:
        using DataList = std::array<Float, N>;
        static constexpr Float AVG_MULTIPLIER = Float(1) / Float(N);

        private:
        DataList    values = {};
        int32_t     index = 0;

        public:
        // Constructors & Destructor
        MovingAverage() = default;
        MovingAverage(Float initialVal) noexcept;

        constexpr void  FeedValue(Float) noexcept;
        constexpr Float Average() const noexcept;
    };

    // TODO: Although not math, put it here
    // Similar to clamp but rolls
    template<SignedIntegralC T> MR_PF_DECL T Roll(T, T minVal, T maxVal) noexcept;

    // Simple Divisions
    template<IntegralC T>       MR_PF_DECL T DivideUp(T value, T divisor) noexcept;
    template<IntegralVectorC T> MR_PF_DECL T DivideUp(const T& value, const T& divisor) noexcept;
    template<IntegralC T>       MR_PF_DECL T NextMultiple(T value, T divisor) noexcept;
    template<IntegralVectorC T> MR_PF_DECL T NextMultiple(const T& value, const T& divisor) noexcept;
    // Misc
    template<IntegralC T> MR_PF_DECL T NextPowerOfTwo(T value) noexcept;
    template<IntegralC T> MR_PF_DECL T PrevPowerOfTwo(T value) noexcept;
    template<IntegralC T> MR_PF_DECL T NextPrime(T value) noexcept;

    // Due to heterogeneous nature of the codebase
    // we wrap all mathemathical functions
    // The reason: https://docs.nvidia.com/cuda/cuda-math-api/index.html
    // Directly quoting NVCC Docs.
    //
    // "Note also that due to implementation constraints, certain math functions
    // from std:: namespace may be callable in device code even via explicitly
    // qualified std:: names. However, such use is discouraged, since this
    // capability is unsupported,  unverified, undocumented, not portable,
    // and may change without notice."
    //
    // Up until now we conditionally pull the std namespace when we call
    // math functions and hope that nvcc etc. resolve to the correct
    // function overload and convert it to intrinsics when available.
    // This becomes an hassle and if we forgot to pull the namespace
    // sometimes host compiler do not find the cmath functions
    // since it is not required to pull these functions to the global namespace.

    // =============== //
    //     COMMON      //
    // =============== //
    template<ArithmeticC T> MR_PF_DECL T Abs(T) noexcept;
    template<ArithmeticC T> MR_PF_DECL T AbsDif(T, T) noexcept;
    template<ArithmeticC T> MR_PF_DECL T Max(T, T) noexcept;
    template<ArithmeticC T> MR_PF_DECL T Min(T, T) noexcept;
    template<ArithmeticC T> MR_PF_DECL T Clamp(T, T min, T max) noexcept;

    template<VectorC T> MR_PF_DECL T Abs(const T&) noexcept;
    template<VectorC T> MR_PF_DECL T AbsDif(const T&, const T&) noexcept;
    template<VectorC T> MR_PF_DECL T Max(const T&, const T&) noexcept;
    template<VectorC T> MR_PF_DECL T Min(const T&, const T&) noexcept;
    template<VectorC T> MR_PF_DECL T Clamp(T, T min, T max) noexcept;
    template<VectorC T> MR_PF_DECL T Clamp(T, typename T::InnerType min,
                                           typename T::InnerType max) noexcept;
    // =============== //
    //   FLOAT ONLY    //
    // =============== //
    // Trigonometry
    template<FloatC T> MR_PF_DECL T Cos(T) noexcept;
    template<FloatC T> MR_PF_DECL T Sin(T) noexcept;
    template<FloatC T> MR_PF_DECL T Tan(T) noexcept;
    template<FloatC T> MR_PF_DECL T CosH(T) noexcept;
    template<FloatC T> MR_PF_DECL T ArcCos(T) noexcept;
    template<FloatC T> MR_PF_DECL T ArcSin(T) noexcept;
    template<FloatC T> MR_PF_DECL T ArcTan(T) noexcept;
    template<FloatC T> MR_PF_DECL T ArcTan2(T, T) noexcept;
    template<FloatC T> MR_PF_DECL T ArcTanH(T) noexcept;
    template<FloatC T> MR_PF_DECL std::array<T, 2> SinCos(T) noexcept;
    // Common functions (math functions)
    template<FloatC T> MR_PF_DECL T ErrFunc(T) noexcept;
    template<FloatC T> MR_PF_DECL T InvErrFunc(T) noexcept;
    template<FloatC T> MR_PF_DECL T Exp(T) noexcept;
    template<FloatC T> MR_PF_DECL T Exp2(T) noexcept;
    template<FloatC T> MR_PF_DECL T Log(T) noexcept;
    template<FloatC T> MR_PF_DECL T Log2(T) noexcept;
    template<FloatC T> MR_PF_DECL T Gaussian(T x, T sigma = T(1), T mu = T(0)) noexcept;
    // Rounding
    template<FloatC T> MR_PF_DECL T Round(T) noexcept;
    template<FloatC T> MR_PF_DECL T Ceil(T) noexcept;
    template<FloatC T> MR_PF_DECL T Floor(T) noexcept;
    template<FloatC T> MR_PF_DECL auto RoundInt(T x) noexcept -> IntegralSister<T>;
    //
    // This pattern comes out too many times in
    // numeric calculations. Due to precision errors,
    // sometimes mathematicall correct square roots
    // may result in NaN.
    // Basically "sqrt(max(0, a))".
    template<FloatC T> MR_PF_DECL T Sqrt(T) noexcept;
    template<FloatC T> MR_PF_DECL T SqrtMax(T) noexcept;
    template<FloatC T> MR_PF_DECL T RSqrt(T) noexcept;
    template<FloatC T> MR_PF_DECL T RSqrtMax(T) noexcept;
    //
    template<FloatC T> MR_PF_DECL T Cbrt(T) noexcept;
    // Graphics-related
    template<FloatC T> MR_PF_DECL T Lerp(T, T, T) noexcept;
    template<FloatC T> MR_PF_DECL T Smoothstep(T, T, T) noexcept;
    template<FloatC T> MR_PF_DECL T InvSmoothstep(T t) noexcept;
    template<FloatC T> MR_PF_DECL T InvSmoothstepApprox(T t) noexcept;
    // Misc.
    template<FloatC T> MR_PF_DECL bool IsNaN(T) noexcept;
    template<FloatC T> MR_PF_DECL bool IsInf(T) noexcept;
    template<FloatC T> MR_PF_DECL bool IsFinite(T) noexcept;
    template<FloatC T> MR_PF_DECL bool SignBit(T) noexcept;
    template<FloatC T> MR_PF_DECL T SignPM1(T) noexcept;
    template<FloatC T> MR_PF_DECL T NextFloat(T) noexcept;
    template<FloatC T> MR_PF_DECL T PrevFloat(T) noexcept;
    // Math-related
    template<FloatC T> MR_PF_DECL T FMA(T, T, T) noexcept;
    template<FloatC T> MR_PF_DECL T FMod(T, T) noexcept;
    template<FloatC T> MR_PF_DECL auto ModF(T) noexcept -> std::array<T, 2>;
    template<FloatC T> MR_PF_DECL auto ModFInt(T) noexcept -> Pair<IntegralSister<T>, T>;
    template<FloatC T> MR_PF_DECL T Pow(T, T) noexcept;
    // For vector types, we add as we needed in generic code,
    // Or friction is good here. When implementing cost of the routine
    // will be present on the caller side.
    template<FloatVectorC T> MR_PF_DECL T Round(const T&) noexcept;
    template<FloatVectorC T> MR_PF_DECL T Ceil(const T&) noexcept;
    template<FloatVectorC T> MR_PF_DECL T Floor(const T&) noexcept;
    template<FloatVectorC T> MR_PF_DECL auto RoundInt(const T&) noexcept -> Vector<T::Dims, IntegralSister<typename T::InnerType>>;
    //
    template<FloatVectorC T> MR_PF_DECL T Sqrt(T) noexcept;
    template<FloatVectorC T> MR_PF_DECL T SqrtMax(T) noexcept;
    template<FloatVectorC T> MR_PF_DECL T RSqrt(T) noexcept;
    template<FloatVectorC T> MR_PF_DECL T RSqrtMax(T) noexcept;
    // Graphics-related
    template<FloatVectorC T> MR_PF_DECL T Lerp(const T&, const T&, typename T::InnerType) noexcept;
    template<FloatVectorC T> MR_PF_DECL T Smoothstep(const T&, const T&, typename T::InnerType) noexcept;
    template<FloatVectorC T> MR_PF_DECL auto Dot(const T&) noexcept -> typename T::InnerType;
    template<FloatVectorC T> MR_PF_DECL auto Length(const T&) noexcept -> typename T::InnerType;
    template<FloatVectorC T> MR_PF_DECL auto LengthSqr(const T&) noexcept -> typename T::InnerType;
    template<FloatVectorC T> MR_PF_DECL T Normalize(const T&) noexcept;
    template<FloatC T>       MR_PF_DECL Vector<3, T> Cross(const Vector<3, T>&, const Vector<3, T>&) noexcept;
    // Misc.
    template<FloatVectorC T> MR_PF_DECL bool IsFinite(const T&) noexcept;
}

namespace Math
{

template<uint32_t N>
inline MovingAverage<N>::MovingAverage(Float initialVal) noexcept
{
    std::fill(values.cbegin(), values.cend(), initialVal);
    index = Roll(index + 1, 0, int32_t(N));
}

template<uint32_t N>
constexpr void MovingAverage<N>::FeedValue(Float v) noexcept
{
    values[static_cast<size_t>(index)] = v;
    index = Roll(index + 1, 0, int32_t(N));
}

template<uint32_t N>
constexpr Float MovingAverage<N>::Average() const noexcept
{
    Float total = std::reduce(values.cbegin(), values.cend(), Float(0));
    return total * AVG_MULTIPLIER;
}

template<SignedIntegralC T>
MR_PF_DEF T Roll(T v, T minVal, T maxVal) noexcept
{
    assert(minVal < maxVal);
    T diff = maxVal - minVal;
    v -= minVal;
    v %= diff;
    v = (v < T(0)) ? diff + v : v ;
    v += minVal;
    return v;
}

// Simple Divisions
template<IntegralC T>
MR_PF_DEF T DivideUp(T value, T divisor) noexcept
{
    assert(divisor != T(0));
    return (value + divisor - 1) / divisor;
}

template<IntegralVectorC T>
MR_PF_DEF T DivideUp(const T& value, const T& divisor) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; ++i)
        r[i] = DivideUp(value[i], divisor[i]);
    return r;
}

template<IntegralC T>
MR_PF_DEF T NextMultiple(T value, T divisor) noexcept
{
    return DivideUp(value, divisor) * divisor;
}

template<IntegralVectorC T>
MR_PF_DEF T NextMultiple(const T& value, const T& divisor) noexcept
{
    return DivideUp(value, divisor) * divisor;
}

template<IntegralC T>
MR_PF_DEF T NextPowerOfTwo(T value) noexcept
{
    assert(value < (T(1) << (std::numeric_limits<T>::digits - 1)));
    if(value <= T(1)) return T(1);
    return T(1) << Bit::RequiredBitsToRepresent(value - 1);
}

template<IntegralC T>
MR_PF_DEF T PrevPowerOfTwo(T value) noexcept
{
    assert(value <= T(1));
    if(value <= T(0)) return T(0);
    return T(1) << (Bit::RequiredBitsToRepresent(value) - 1);
}

template<IntegralC T>
MR_PF_DEF T NextPrime(T value) noexcept
{
    constexpr std::array<T, 64> FIRST_PRIMES =
    {
          2,   3,   5,  7,   11,  13,  17,  19,
         23,  29,  31,  37,  41,  43,  47,  53,
         59,  61,  67,  71,  73,  79,  83,  89,
         97, 101, 103, 107, 109, 113, 127, 131,
        137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223,
        227, 229, 233, 239, 241, 251, 257, 263,
        269, 271, 277, 281, 283, 293, 307, 311
    };

    auto IsPrime = [&FIRST_PRIMES](T v)
    {
        for(T prime : FIRST_PRIMES)
            if(v % prime == 0) return false;

        // Continue traversing incrementally now
        for(uint32_t i = FIRST_PRIMES[63] + 2; (i * i) < v; i += 2)
            if(v % i == 0) return false;

        return true;
    };
    if(value <= FIRST_PRIMES[63])
    {
        return *std::lower_bound(FIRST_PRIMES.cbegin(),
                                 FIRST_PRIMES.cend(),
                                 value);
    }
    // Check one by one
    for(T i = value;; i++)
    {
        if(IsPrime(i)) return i;
    }
}

template<ArithmeticC T>
MR_PF_DEF T Abs(T v) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Abs is not constexpr until C++23
        // Easy part unsigned integral
        if constexpr(std::is_unsigned_v<T>) return v;
        else                                return (v < T(0)) ? (-v) : (v);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::abs(v);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)      return fabsf(v);
        if constexpr(std::is_same_v<T, double>)     return fabs(v);
        if constexpr(std::is_same_v<T, int64_t>)    return llabs(v);
        if constexpr(std::is_same_v<T, int32_t>)    return abs(v);
        // For small types lift to the 32-bit (most native to GPU
        // and least register usage)
        if constexpr(std::is_signed_v<T>)           return T(abs(int32_t(v)));
        // Unsigned integral types left...
        return v;
    #endif
}

template<ArithmeticC T>
MR_PF_DEF T AbsDif(T x, T y) noexcept
{
    if(std::is_constant_evaluated())
        return (x > y) ? (x - y) : (y - x);

    #ifndef MRAY_DEVICE_CODE_PATH
        if constexpr(std::is_floating_point_v<T>)
            return std::fdim(x, y);
        // Integral
        else return (x > y) ? (x - y) : (y - x);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)
            return fdimf(x, y);
        if constexpr(std::is_same_v<T, double>)
            return fdim(x, y);
        else return (x > y) ? (x - y) : (y - x);
    #endif
}

template<ArithmeticC T>
MR_PF_DEF T Max(T x, T y) noexcept
{
    if(std::is_constant_evaluated())
        return std::max(x, y);
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::max(x, y);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)   return fmaxf(x, y);
        if constexpr(std::is_same_v<T, double>)  return fmax(x, y);
        if constexpr(std::is_same_v<T, int64_t>) return max(x, y);
        if constexpr(std::is_same_v<T, int32_t>) return max(x, y);
        // For small types lift to the 32-bit (most native to GPU
        // and least register usage)
        if constexpr(std::is_signed_v<T>)        return T(max(int32_t(x), int32_t(y)));
        if constexpr(std::is_unsigned_v<T>)      return T(max(uint32_t(x), uint32_t(y)));
    #endif
}

template<ArithmeticC T>
MR_PF_DEF T Min(T x, T y) noexcept
{
    if(std::is_constant_evaluated())
        return std::min(x, y);
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::min(x, y);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)   return fminf(x, y);
        if constexpr(std::is_same_v<T, double>)  return fmin(x, y);
        if constexpr(std::is_same_v<T, int64_t>) return min(x, y);
        if constexpr(std::is_same_v<T, int32_t>) return min(x, y);
        // For small types lift to the 32-bit (most native to GPU
        // and least register usage)
        if constexpr(std::is_signed_v<T>)        return T(min(int32_t(x), int32_t(y)));
        if constexpr(std::is_unsigned_v<T>)      return T(min(uint32_t(x), uint32_t(y)));
    #endif
}

template<ArithmeticC T>
MR_PF_DEF T Clamp(T v, T min, T max) noexcept
{
    return Min(Max(min, v), max);
}

template<VectorC T>
MR_PF_DEF T Abs(const T& v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; ++i)
        r[i] = Abs(v[i]);
    return r;
}

template<VectorC T>
MR_PF_DEF T AbsDif(const T& v0, const T& v1) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
        for(unsigned int i = 0; i < N; ++i)
            r[i] = AbsDif(v0[i], v1[i]);
    return r;
}

template<VectorC T>
MR_PF_DEF T Max(const T& v0, const T& v1) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; ++i)
        r[i] = Max(v0[i], v1[i]);
    return r;
}

template<VectorC T>
MR_PF_DEF T Min(const T& v0, const T& v1) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; ++i)
        r[i] = Min(v0[i], v1[i]);
    return r;
}

template<VectorC T>
MR_PF_DEF T Clamp(T v, T min, T max) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; ++i)
        r[i] = Clamp(v[i], min[i], max[i]);
    return r;
}

template<VectorC T>
MR_PF_DEF T Clamp(T v, typename T::InnerType min,
                  typename T::InnerType max) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; ++i)
        r[i] = Clamp(v[i], min, max);
    return r;
}

template<FloatC T>
MR_PF_DEF T Cos(T v) noexcept
{
    if(std::is_constant_evaluated())
    {
        // TODO: Check literature for this?
        // First I was going to do tylor series but found
        // a njuffa gem "https://stackoverflow.com/a/63931958".

        // Move to [0, pi] as the function requires
        // Constexpr has no "modf" so manually calculate
        using D = double;
        double ldV = static_cast<D>(v);
        constexpr D edge = (MathConstants::Pi<D>() * D(2));
        D r = ldV / edge;
        r = static_cast<D>(static_cast<int64_t>(r));
        D x = ldV - r * edge;
        // cos(-x) = -cos(x)
        bool isLarge = x > MathConstants::Pi<D>();
        if(isLarge) x = x - MathConstants::Pi<D>();

        const D half_pi_hi       =  1.57079637e+0; //  0x1.921fb6p+0
        const D half_pi_lo       = -4.37113883e-8; // -0x1.777a5cp-25
        const D three_half_pi_hi =  4.71238899e+0; //  0x1.2d97c8p+2
        const D three_half_pi_lo = -1.19248806e-8; // -0x1.99bc5cp-27
        // cos(x) = sin (pi/2 - x) = sin (hpmx)
        D hpmx = (half_pi_hi - x) + half_pi_lo;               // pi/2 - x
        D thpmx = (three_half_pi_hi - x) + three_half_pi_lo;  // 3*pi/2 - x
        D nhpmx = (-half_pi_hi - x) - half_pi_lo;             // -pi/2 - x

        // P(hpmx*hpmx) ~= sin (hpmx) / (hpmx * (hpmx * hpmx - pi * pi))
        D p, s;
        s = hpmx * hpmx;
        p =           D(+1.32823530e-10); //  0x1.241500p-33
        p = FMA(p, s, D(-2.33173445e-8)); // -0x1.9096c4p-26
        p = FMA(p, s, D(+2.52237896e-6)); //  0x1.528c48p-19
        p = FMA(p, s, D(-1.73501656e-4)); // -0x1.6bdbfep-13
        p = FMA(p, s, D(+6.62087509e-3)); //  0x1.b1e7dap-8
        p = FMA(p, s, D(-1.01321183e-1)); // -0x1.9f02f6p-4
        return T((isLarge ? D(-1) : D(1)) * hpmx * nhpmx * thpmx * p);
        //
        // Commented old code
        //// Evaluating first 7 terms
        //LD t7 = ((x / 12) * (x / 11) * (x / 10) * (x / 9) *
        //         (x / 8) * (x / 7) * (x / 6) * (x / 5) *
        //         (x / 4) * (x / 3) * (x / 2) * (x));
        //LD t6 = ((x / 10) * (x / 9) * (x / 8) * (x / 7) * (x / 6) *
        //         (x / 5) * (x / 4) * (x / 3) * (x / 2) * x);
        //LD t5 = ((x / 8) * (x / 7) * (x / 6) * (x / 5) *
        //         (x / 4) * (x / 3) * (x / 2) * (x));
        //LD t4 = ((x / 6) * (x / 5) * (x / 4) * (x / 3) *
        //         (x / 2) * (x);
        //LD t3 = ((x / 4) * (x / 3) * (x / 2) * x);
        //LD t2 = ((x / 2) * x);
        //return T(LD(1) + (t3 - t2) + (t5 - t4) + (t7 - t6));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::cos(v);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return cosf(v);
        if constexpr(std::is_same_v<T, double>) return cos(v);
    #endif
}

template<FloatC T>
MR_PF_DEF T Sin(T v) noexcept
{
    if(std::is_constant_evaluated())
    {
        using MathConstants::Pi;
        return Math::Cos<Float>(Pi<Float>() * Float(0.5) - v);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::sin(v);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return sinf(v);
        if constexpr(std::is_same_v<T, double>) return sin(v);
    #endif
}

template<FloatC T>
MR_PF_DEF T Tan(T v) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Another gem from njuffa
        // https://stackoverflow.com/questions/77536578/tanx-approximation-to-double-precision-on-x-pi-4
        // This time we move to the [-90, 90] range
        using D = double;
        D ldV = static_cast<D>(v);
        constexpr D edge = (MathConstants::Pi<D>() * D(0.5));
        D roll = ldV / edge;
        roll = static_cast<D>(static_cast<int64_t>(roll));
        D x = ldV - roll * edge;
        //static_assert(x <= (MathConstants::Pi<D>() * D(0.5)) &&
        //              x >= (-MathConstants::Pi<D>() * D(0.5)));
        //return x * MathConstants::RadToDegCoef<T>();

        D s, p, q, r;
        s = x * x;
        p =           D(+6.0284853818520106e-4);  //  0x1.3c10f60c2ddc2p-11
        p = FMA(p, s, D(-6.1615805421222844e-2)); // -0x1.f8c1b5a66d606p-5
        p = FMA(p, s, D(+1.0000000000000000e+0)); //  0x1.0000000000000p+0
        q =           D(-6.2529620840320529e-4);  // -0x1.47d5d61f8c6e0p-11
        q = FMA(q, s, D(+7.0033226411156085e-2)); //  0x1.1edb29111bc42p-4
        q = FMA(q, s, D(-1.3848474162642035e+0)); // -0x1.62855c3ace0adp+0
        q = FMA(q, s, D(+3.0000000000000178e+0)); //  0x1.8000000000028p+1
        r = FMA(p / q, s * x, x);
        return T(r);
        // This was not good on the edges (x ~= +-90)
        // Prob due to catastrophic cancellation
        // return Math::Sin<Float>(v) / Math::Cos<Float>(v);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::tan(v);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return tanf(v);
        if constexpr(std::is_same_v<T, double>) return tan(v);
    #endif
}

template<FloatC T>
MR_PF_DECL T CosH(T x) noexcept
{
    // no njuffa :(
    // Using mathematical identity
    if(std::is_constant_evaluated())
    {
        T ex = Exp(-x);
        T e2x = ex * ex;
        return Float(1) + e2x / (Float(0.5) * ex);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::cosh(x);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return coshf(x);
        if constexpr(std::is_same_v<T, double>) return cosh(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T ArcCos(T x) noexcept
{
    assert(T(-1) <= x && x <= T(1));
    // njuffa
    // https://stackoverflow.com/a/7380529
    if(std::is_constant_evaluated())
    {
        return ArcTan2(Sqrt(T(1) + x) * (T(1) - x), x);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::acos(x);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return acosf(x);
        if constexpr(std::is_same_v<T, double>) return acos(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T ArcSin(T x) noexcept
{
    assert(T(-1) <= x && x <= T(1));
    // njuffa
    // https://stackoverflow.com/a/7380529
    if(std::is_constant_evaluated())
    {
        return ArcTan2(x, Sqrt(T(1) + x) * (T(1) - x));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::asin(x);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return asinf(x);
        if constexpr(std::is_same_v<T, double>) return asin(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T ArcTan(T x) noexcept
{
    // njuffa
    // https://forums.developer.nvidia.com/t/weekend-project-fast-arctangent-computation-with-borchardts-algorithm/303546
    if(std::is_constant_evaluated())
    {
        constexpr T pio2 = T(1.57079633);
        T t = FMA(x, x, T(1));
        T b0 = Sqrt(t);
        t = T(0.5) * t;
        t = FMA(T(0.5), b0, t);
        T a1 = FMA(T(0.5), b0, T(0.5));
        T b1 = Sqrt(t);
        t = FMA(a1, b1, t) * T(0.5);
        T b2 = Sqrt(t);
        T r;
        r = FMA(b0, T(0.076905564f), T(0.076589905f));  // 0x1.3b0154p-4, 0x1.39b656p-4
        r = FMA(b1, T(0.124636024f), r);                // 0x1.fe8258p-4
        r = FMA(b2, T(0.721868575f), r);                // 0x1.7198c2p-1
        r = T(1) / r;
        r = x * r;
        if(Abs(x) >= T(0x1.0p64)) r = std::copysign(pio2, x); // may be optional
        return r;
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::atan(x);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return atanf(x);
        if constexpr(std::is_same_v<T, double>) return atan(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T ArcTan2(T y, T x) noexcept
{
    // njuffa
    // https://stackoverflow.com/a/46251566
    if(std::is_constant_evaluated())
    {
        T ax = Abs(x);
        T ay = Abs(y);
        T mx = Max(ay, ax);
        T mn = Min(ay, ax);
        T a = mn / mx;
        // Minimax polynomial approximation to atan(a) on [0,1]
        T s = a * a;
        T c = s * a;
        T q = s * s;
        T r = FMA(T(+0.024840285), q, T(+0.18681418));
        T t = FMA(T(-0.094097948), q, T(-0.33213072));
        r = r * s + t;
        r = r * c + a;
        // Map to full circle
        if(ay > ax) r = T(1.57079637) - r;
        if(x <   0) r = T(3.14159274) - r;
        if(y <   0) r = -r;
        return r;
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::atan2(y, x);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return atan2f(y, x);
        if constexpr(std::is_same_v<T, double>) return atan2(y, x);
    #endif
}

template<FloatC T> MR_PF_DECL T ArcTanH(T x) noexcept
{
    // Using math identity, TODO: check better one later.
    if(std::is_constant_evaluated())
    {
        return Float(0.5) * Log((Float(1) + x) / (Float(1) - x));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::atanh(x);
    #else
        if constexpr(std::is_same_v<T, float>)  return atanhf(x);
        if constexpr(std::is_same_v<T, double>) return atanh(x);
    #endif
}

template<FloatC T>
MR_PF_DEF std::array<T, 2> SinCos(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        return {Math::Sin(x), Math::Cos(x)};
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return {Math::Sin(x), Math::Cos(x)};
    #else
        std::array<T, 2> r;
        if constexpr(std::is_same_v<T, float>)  sincosf(x, r.data() + 0, r.data() + 1);
        if constexpr(std::is_same_v<T, double>) sincos(x, r.data() + 0, r.data() + 1);
        return r;
    #endif
}

template<FloatC T>
MR_PF_DEF T ErrFunc(T x) noexcept
{
    // njuffa
    // https://forums.developer.nvidia.com/t/calling-all-juffas-whats-up-with-erfcf-nowadays/262973/3
    if(std::is_constant_evaluated())
    {
        using F = float;
        F t;
        t = -1.64611265e-6f;            // -0x1.b9e000p-20
        t = FMA(t, x, 2.95254722e-5f);  //  0x1.ef5af0p-16
        t = FMA(t, x, -2.33422339e-4f); // -0x1.e985aap-13
        t = FMA(t, x, 1.04246172e-3f);  //  0x1.11466cp-10
        t = FMA(t, x, -2.55015842e-3f); // -0x1.4e411ep-9
        t = FMA(t, x, 3.19798535e-4f);  //  0x1.4f5544p-12
        t = FMA(t, x, 2.76054665e-2f);  //  0x1.c449b8p-6
        t = FMA(t, x, -1.48274124e-1f); // -0x1.2faa58p-3
        t = FMA(t, x, -9.18447673e-1f); // -0x1.d63ec6p-1
        t = FMA(t, x, -1.62790680e+0f); // -0x1.a0be80p+0
        t = t * x;
        return T(Exp2<F>(t));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::erf(x);
    #else
        if constexpr(std::is_same_v<T, float>)  return erff(x);
        if constexpr(std::is_same_v<T, double>) return erf(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T InvErrFunc(T x) noexcept
{
    #ifdef MRAY_DEVICE_CODE_PATH
        if constexpr(std::is_same_v<T, float>)  return erfinvf(x);
        if constexpr(std::is_same_v<T, double>) return erfinv(x);
    #endif
    // Checked the pbrt-v4, it has similar impl
    // of this (From a stackoverflow post).
    // https://stackoverflow.com/a/49743348
    //
    // https://people.maths.ox.ac.uk/gilesm/codes/erfinv/gems.pdf
    // I've checked other sites and find this
    // http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function
    // However could not implement it properly (numerical precision errors)
    // Using the stackoverflow one
    //
    T t = Log(FMA(x, T(0) - x, T(1)));
    T p ;
    if(Abs(t) > T(6.125))
    {
        p =           T(+3.03697567e-10);
        p = FMA(p, t, T(+2.93243101e-8));
        p = FMA(p, t, T(+1.22150334e-6));
        p = FMA(p, t, T(+2.84108955e-5));
        p = FMA(p, t, T(+3.93552968e-4));
        p = FMA(p, t, T(+3.02698812e-3));
        p = FMA(p, t, T(+4.83185798e-3));
        p = FMA(p, t, T(-2.64646143e-1));
        p = FMA(p, t, T(+8.40016484e-1));
    }
    else
    {
        p =           T(5.438778320e-9);
        p = FMA(p, t, T(+1.43285448e-7));
        p = FMA(p, t, T(+1.22774793e-6));
        p = FMA(p, t, T(+1.12963626e-7));
        p = FMA(p, t, T(-5.61530760e-5));
        p = FMA(p, t, T(-1.47697632e-4));
        p = FMA(p, t, T(+2.31468678e-3));
        p = FMA(p, t, T(+1.15392581e-2));
        p = FMA(p, t, T(-2.32015476e-1));
        p = FMA(p, t, T(+8.86226892e-1));
    }
    T r = x * p;
    return r;
}

template<FloatC T>
MR_PF_DEF T Exp(T x) noexcept
{
    // Another njuffa solution
    // His name was on the license "Norbert Juffa"
    // https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528
    if(std::is_constant_evaluated())
    {
        // This function is designed for floats
        // We cant use T (it can be double)
        using F = float;
        using I = IntegralSister<float>;
        F f, r, j, s, t;
        // exp(a) = 2**i * exp(f); i = rintf(a / log(2))
        j = FMA(1.442695f, x, 12582912.f);
        j -= 12582912.f;                    // 0x1.715476p0, 0x1.8p23
        f = FMA(j, -6.93145752e-1f, x);     // -0x1.62e400p-1  // log_2_hi
        f = FMA(j, -1.42860677e-6f, f);     // -0x1.7f7d1cp-20 // log_2_lo
        // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
        r = 1.37805939e-3f;                 // 0x1.694000p-10
        r = FMA(r, f, 8.37312452e-3f);      // 0x1.125edcp-7
        r = FMA(r, f, 4.16695364e-2f);      // 0x1.555b5ap-5
        r = FMA(r, f, 1.66664720e-1f);      // 0x1.555450p-3
        r = FMA(r, f, 4.99999851e-1f);      // 0x1.fffff6p-2
        r = FMA(r, f, 1.00000000e+0f);      // 0x1.000000p+0
        r = FMA(r, f, 1.00000000e+0f);      // 0x1.000000p+0
        // exp(a) = 2**i * r
        I i = static_cast<I>(j);
        I ix = (i > 0) ? 0 : std::bit_cast<I>(0x83000000);
        s = Bit::BitCast<F>(0x7f000000 + ix);
        t = Bit::BitCast<F>((i << 23) - ix);
        r = r * s;
        r = r * t;
        // handle special cases: severe overflow / underflow
        if(Abs(x) >= 104.0f) r = s * s;
        return r;
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::exp(x);
    #else
        if constexpr(std::is_same_v<T, float>)  return expf(x);
        if constexpr(std::is_same_v<T, double>) return exp(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Exp2(T x) noexcept
{
    // Another njuffa solution
    // https://stackoverflow.com/a/65562273
    if(std::is_constant_evaluated())
    {
        using D = double;
        using I = IntegralSister<D>;
        constexpr D FP64_MIN_EXPO = -1022; // exponent of minimum binary64 normal
        constexpr I FP64_MANT_BITS = 52;   // number of stored mantissa (significand) bits
        constexpr D FP64_EXPO_BIAS = 1023; // binary64 exponent bias
        D p = (D(x) < FP64_MIN_EXPO) ? FP64_MIN_EXPO : D(x); // clamp below
        // 2**p = 2**(w+z), with w an integer and z in [0, 1)
        D w = Floor(p); // integral part
        D z = D(x) - w;     // fractional part
        // approximate 2**z-1 for z in [0, 1)
        D approx = -0x1.6e75d58p+2 + 0x1.bba7414p+4;
        approx /= (0x1.35eccbap+2 - z) - 0x1.f5e53c2p-2 * z;
        // assemble the exponent and mantissa components into final result
        I resI = (I(1) << FP64_MANT_BITS) * I(w + FP64_EXPO_BIAS + approx);
        return T(Bit::BitCast<D>(resI));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::exp2(x);
    #else
        if constexpr(std::is_same_v<T, float>)  return exp2f(x);
        if constexpr(std::is_same_v<T, double>) return exp2(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Log(T x) noexcept
{
    // Another njuffa solution
    // https://stackoverflow.com/a/39822314
    if(std::is_constant_evaluated())
    {
        using F = float;
        using I = IntegralSister<F>;
        I e = (Bit::BitCast<I>(x) - I(0x3F2AAAAB)) & I(0xFF800000);
        F m = Bit::BitCast<F>(Bit::BitCast<I>(x) - e);
        F i = F(e) * 1.19209290e-7f; // 0x1.0p-23
        // m in [2/3, 4/3]
        F f = m - 1.0f;
        F s = f * f;
        // Compute log1p(f) for f in [-1/3, 1/3]
        F r = FMA(0.230836749f, f, -0.279208571f); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        F t = FMA(0.331826031f, f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2
        r = FMA(r, s, t);
        r = FMA(r, s, f);
        r = FMA(i, 0.693147182f, r); // 0x1.62e430p-1 // log(2)
        return T(r);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::log(x);
    #else
        if constexpr(std::is_same_v<T, float>)  return logf(x);
        if constexpr(std::is_same_v<T, double>) return log(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Log2(T x) noexcept
{
    // Guess what
    // https://forums.developer.nvidia.com/t/faster-and-more-accurate-implementation-of-log2f/40635
    if(std::is_constant_evaluated())
    {
        using F = float;
        using I = IntegralSister<F>;
        // Some bit manipulation is better with uints
        // less undefined behaviour
        using UI = std::make_unsigned_t<I>;
        F m, r;
        F i = 0.0f;
        if (x < 1.175494351e-38f)   // 0x1.0p-126
        {
            x = x * 8388608.0f;     // 0x1.0p+23
            i = -23.0f;
        }
        I e = I((Bit::BitCast<UI>(x) - UI(0x3f3504f3)) & UI(0xff800000));
        m = Bit::BitCast<F>(Bit::BitCast<I>(x) - e);
        i = FMA(float(e), 1.19209290e-7f, i); // 0x1.0p-23
        m = m - 1.0f;
        // Compute log2(1+m) for m in [sqrt(0.5)-1, sqrt(2.0)-1]
        r =           -1.09985352e-1f;  // -0x1.c28000p-4
        r = FMA (r, m, 1.86182275e-1f); //  0x1.7d4d22p-3
        r = FMA(r, m, -1.91066533e-1f); // -0x1.874de4p-3
        r = FMA(r, m,  2.04593703e-1f); //  0x1.a30206p-3
        r = FMA(r, m, -2.39627063e-1f); // -0x1.eac198p-3
        r = FMA(r, m,  2.88573444e-1f); //  0x1.277fccp-2
        r = FMA(r, m, -3.60695332e-1f); // -0x1.715a1ep-2
        r = FMA(r, m,  4.80897635e-1f); //  0x1.ec706ep-2
        r = FMA(r, m, -7.21347392e-1f); // -0x1.715472p-1
        r = FMA(r, m,  4.42695051e-1f); //  0x1.c551dap-2
        r = FMA(r, m, m);
        r = r + i;
        // Check for and handle special cases
        constexpr F Inf = Bit::BitCast<F>(0x7f800000); // +INF
        // We cant use inline assembly in constexpr.
        // This is one place that is different from the link above.
        if(x <= 0.0f)   return -std::numeric_limits<F>::infinity();
        if(x > Inf)     return std::numeric_limits<F>::infinity();
        else            return T(r);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::log2(x);
    #else
        if constexpr(std::is_same_v<T, float>)  return log2f(x);
        if constexpr(std::is_same_v<T, double>) return log2(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Gaussian(T x, T sigma, T mu) noexcept
{
    assert(sigma > 0);
    using namespace MathConstants;
    constexpr T InvSqrt2Pi = (T(1) / Sqrt2<T>()) *  (T(1) / SqrtPi<T>());
    T sigmaInv = T(1) / sigma;
    T result = InvSqrt2Pi * sigmaInv;
    T pow = (x - mu) * sigmaInv;
    result *= Math::Exp(T(-0.5) * pow * pow);
    return result;
}

template<FloatC T>
MR_PF_DEF T Round(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Not good, but it is constexpr
        // so no undefined behavour is allowed
        // we can get sloppy code.
        using I = IntegralSister<T>;
        if(x < T(0)) return T(I(x - NextFloat(T(0.5))));
        else         return T(I(x + NextFloat(T(0.5))));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::round(x);
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return roundf(x);
        if constexpr(std::is_same_v<T, double>) return round(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Ceil(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Not good, but it is constexpr
        // so no undefined behavour is allowed
        // we can get sloppy code.
        // Also for large numbers this will shit the bed
        //
        using I = IntegralSister<T>;
        if(x < T(0)) return T(I(x - PrevFloat<T>(1)));
        else         return T(I(x + PrevFloat<T>(1)));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::ceil(x);
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return ceilf(x);
        if constexpr(std::is_same_v<T, double>) return ceil(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Floor(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Not good, but it is constexpr
        // so no undefined behavour is allowed
        // we can get sloppy code.
        // Also for large numbers this will shit the bed
        //
        using I = IntegralSister<T>;
        return T(I(x));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::floor(x);
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return floorf(x);
        if constexpr(std::is_same_v<T, double>) return floor(x);
    #endif
}

template<FloatC T>
MR_PF_DEF auto RoundInt(T x) noexcept -> IntegralSister<T>
{
    if(std::is_constant_evaluated())
    {
        return IntegralSister<T>(Round(x));
    }
    using I = IntegralSister<T>;
    #ifndef MRAY_DEVICE_CODE_PATH
        if constexpr(std::is_same_v<T, float>)  return I(std::lround(x));
        else                                    return I(std::llround(x));
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return I(lroundf(x));
        if constexpr(std::is_same_v<T, double>) return I(llround(x));
    #endif
}

template<FloatC T>
MR_PF_DEF T Sqrt(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Couldn't find njuffa solution :(
        // Doing classic Newton-Raphson
        // https://math.mit.edu/~stevenj/18.335/newton-sqrt.pdf
        auto NRIteration = [&x](T g) -> T
        {
            return T(0.5) * (g + x / g);
        };
        if(x < 0) return std::numeric_limits<T>::quiet_NaN();
        //
        T g = x / 4;
        g = NRIteration(g);
        g = NRIteration(g);
        g = NRIteration(g);
        g = NRIteration(g);
        g = NRIteration(g);
        return g;
    }
    assert(x >= T(0));
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::sqrt(x);
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return sqrtf(x);
        if constexpr(std::is_same_v<T, double>) return sqrt(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T SqrtMax(T x) noexcept
{
    return Sqrt(Max(x, T(0)));
}

template<FloatC T>
MR_PF_DEF T RSqrt(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        return T(1) / Sqrt(x);
    }
    assert(x >= T(0));
    #ifndef MRAY_DEVICE_CODE_PATH
        return T(1) / Sqrt(x);
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return rsqrtf(x);
        if constexpr(std::is_same_v<T, double>) return rsqrt(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T RSqrtMax(T x) noexcept
{
    return RSqrt(Max(x, T(0)));
}

template<FloatC T>
MR_PF_DEF T Cbrt(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        // TODO: Not accurate..
        // https://stackoverflow.com/questions/18063755/computing-a-correctly-rounded-an-almost-correctly-rounded-floating-point-cubic
        // Implement the njuffa here
        return Pow(x, T(0.333333333333));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::cbrt(x);
    #else
        //
        if constexpr(std::is_same_v<T, float>)  return cbrtf(x);
        if constexpr(std::is_same_v<T, double>) return cbrt(x);
    #endif
}

template<FloatC T>
MR_PF_DEF T Lerp(T a, T b, T t) noexcept
{
    assert(t >= T(0) && t <= T(1));
    return a * (T{1} - t) + b * t;
}

template<FloatC T>
MR_PF_DEF T Smoothstep(T a, T b, T t) noexcept
{
    assert(t >= T(0) && t <= T(1));
    // https://en.wikipedia.org/wiki/Smoothstep
    t = Clamp((t - a) / (b - a), T{0}, T{1});
    return t * t * (T{3} - T{2} * t);
}

template<FloatC T>
MR_PF_DEF T InvSmoothstep(T y) noexcept
{
    // It seems Smoothstep has analytic solution for its inverse.
    // https://iquilezles.org/articles/ismoothstep/
    assert(y >= T(0) && y <= T(1));
    constexpr T FACTOR = T(1) / T(3);
    return T(0.5) - Math::Sin(Math::ArcSin(T(1) - T(2) * y) * FACTOR);
}

template<FloatC T>
MR_PF_DEF T InvSmoothstepApprox(T y) noexcept
{
    // Approximation does not give 0 and 1 directly,
    // (function derivates at edges)
    // this may be a problem, so we manually clamp here
    if(y < MathConstants::Epsilon<T>()) return T(0);
    if(y > T(1) - MathConstants::Epsilon) return T(1);

    // Quite nice approximation,
    // https://iradicator.com/fast-inverse-smoothstep/
    assert(y >= T(0) && y <= T(1));
    T yn = T(2) * y - T(1);
    T absyn3 = Math::Abs(yn) * yn * yn;
    T t = T(0.45) * yn + T(0.5) * yn * (absyn3 * absyn3 - T(0.9) * absyn3);
    t -= (t * (T(4) * t * t - T(3)) + yn) / (T(12) * t * t - T(3));
    // Second descend probably needed
    t -= (t * (T(4) * t * t - T(3)) + yn) / (T(12) * t * t - T(3));
    return t + T(0.5);
}

template<FloatC T>
MR_PF_DEF bool IsNaN(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        return (x != x);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::isnan(x);
    #else
        return isnan(x);
    #endif
}

template<FloatC T>
MR_PF_DEF bool IsInf(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        return (x == std::numeric_limits<T>::infinity() ||
                x == -std::numeric_limits<T>::infinity());
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::isinf(x);
    #else
        return isinf(x);
    #endif
}

template<FloatC T>
MR_PF_DEF bool IsFinite(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        return !(IsNaN(x) || IsInf(x));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::isfinite(x);
    #else
        return isfinite(x);
    #endif
}

template<FloatC T>
MR_PF_DEF bool SignBit(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        using I = std::make_unsigned_t<IntegralSister<T>>;
        I v = Bit::BitCast<I>(x);
        I bit = v & (I(1) << (std::numeric_limits<I>::digits - 1));
        return (bit == I(0));
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::signbit(x);
    #else
        return signbit(x) != 0;
    #endif
}

template<FloatC T>
MR_PF_DEF T SignPM1(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        return SignBit(x) ? T(-1) : T(1);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::copysign(T(1), x);
    #else
        if constexpr(std::is_same_v<T, float>)  return copysignf(T(1), x);
        if constexpr(std::is_same_v<T, double>) return copysign(T(1), x);
    #endif
}

template<FloatC T>
MR_PF_DEF T NextFloat(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        using I = std::make_unsigned_t<IntegralSister<T>>;
        I v = Bit::BitCast<I>(x);
        return Bit::BitCast<T>(++v);
    }
    constexpr T MAX = std::numeric_limits<T>::max();
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::nextafter(x, MAX);
    #else
        if constexpr(std::is_same_v<T, float>)  return nextafterf(x, MAX);
        if constexpr(std::is_same_v<T, double>) return nextafter(x, MAX);
    #endif
}

template<FloatC T>
MR_PF_DEF T PrevFloat(T x) noexcept
{
    if(std::is_constant_evaluated())
    {
        using I = std::make_unsigned_t<IntegralSister<T>>;
        I v = Bit::BitCast<I>(x);
        return Bit::BitCast<T>(--v);
    }
    constexpr T MIN = -std::numeric_limits<T>::max();
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::nextafter(x, MIN);
    #else
        if constexpr(std::is_same_v<T, float>)  return nextafterf(x, MIN);
        if constexpr(std::is_same_v<T, double>) return nextafter(x, MIN);
    #endif
}

template<FloatC T>
MR_PF_DEF T FMA(T a, T b, T c) noexcept
{
    if(std::is_constant_evaluated())
    {
        return a * b + c;
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::fma(a, b, c);
    // GPU Code path
    #else
        if constexpr(std::is_same_v<T, float>)  return fmaf(a, b, c);
        if constexpr(std::is_same_v<T, double>) return fma(a, b, c);
    #endif
}

template<FloatC T>
MR_PF_DEF T FMod(T x, T y) noexcept
{
    if(std::is_constant_evaluated())
    {
        return x - y * Floor(x / y);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::fmod(x, y);
    #else
        if constexpr(std::is_same_v<T, float>)  return fmodf(x, y);
        if constexpr(std::is_same_v<T, double>) return fmod(x, y);
    #endif
}

template<FloatC T>
MR_PF_DEF auto ModF(T x) noexcept -> std::array<T, 2>
{
    if(std::is_constant_evaluated())
    {
        T intPart = Math::Ceil(x);
        T fracPart = x - intPart;
        return {intPart, fracPart};
    }
    T intPart, fracPart;
    #ifndef MRAY_DEVICE_CODE_PATH
        fracPart = std::modf(x, &intPart);
    #else
        if constexpr(std::is_same_v<T, float>)  fracPart = modff(x, &intPart);
        if constexpr(std::is_same_v<T, double>) fracPart = modf(x, &intPart);
    #endif
    return {intPart, fracPart};
}

template<FloatC T>
MR_PF_DEF auto ModFInt(T x) noexcept -> Pair<IntegralSister<T>, T>
{
    const auto& [i, f] = ModF(x);
    return Pair(static_cast<IntegralSister<T>>(i), T(f));
}

template<FloatC T>
MR_PF_DECL T Pow(T x, T y) noexcept
{
    if(std::is_constant_evaluated())
    {
        // Invoke undefined behaviour since it is not yet implemented
        //return T(std::numeric_limits<int32_t>::max() + 1);
        return T(0);
    }
    #ifndef MRAY_DEVICE_CODE_PATH
        return std::pow(x, y);
    #else
        if constexpr(std::is_same_v<T, float>)  return powf(x, y);
        if constexpr(std::is_same_v<T, double>) return pow(x, y);
    #endif
}

template<FloatVectorC T>
MR_PF_DEF T Round(const T& v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = Round(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T Ceil(const T& v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = Ceil(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T Floor(const T& v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = Floor(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DECL auto RoundInt(const T& v) noexcept -> Vector<T::Dims, IntegralSister<typename T::InnerType>>
{
    using VecXInt = Vector<T::Dims, IntegralSister<typename T::InnerType>>;
    VecXInt r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = RoundInt(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T Sqrt(T v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = Sqrt(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T SqrtMax(T v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = SqrtMax(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T RSqrt(T v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = RSqrt(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T RSqrtMax(T v) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = RSqrtMax(v[i]);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T Lerp(const T& a, const T& b, typename T::InnerType t) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = Lerp(a[i], b[i], t);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF T Smoothstep(const T& a, const T& b, typename T::InnerType t) noexcept
{
    T r;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r[i] = Smoothstep(a[i], b[i], t);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF auto Dot(const T& v0, const T& v1) noexcept -> typename T::InnerType
{
    typename T::InnerType r = 0;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        r = FMA(v0[i], v1[i], r);
    }
    return r;
}

template<FloatVectorC T>
MR_PF_DEF auto Length(const T& v) noexcept -> typename T::InnerType
{
    #ifdef MRAY_DEVICE_CODE_PATH
        constexpr unsigned int N = T::Dims;
        using F = typename T::InnerType;
        if constexpr(N == 3 && std::is_same_v<F, float>)  return norm3df(v[0], v[1], v[2]);
        if constexpr(N == 4 && std::is_same_v<F, float>)  return norm4df(v[0], v[1], v[2], v[3]);
        if constexpr(N == 3 && std::is_same_v<F, double>) return norm3d(v[0], v[1], v[2]);
        if constexpr(N == 4 && std::is_same_v<F, double>) return norm4d(v[0], v[1], v[2], v[3]);
    #endif
    return Sqrt(LengthSqr(v));
}

template<FloatVectorC T>
MR_PF_DEF auto LengthSqr(const T& v) noexcept -> typename T::InnerType
{
    return Dot(v, v);
}

template<FloatVectorC T>
MR_PF_DEF T Normalize(const T& v) noexcept
{
    using F = typename T::InnerType;
    F lRecip = Float(1) / Length(v);
    return v * lRecip;
}

template<FloatC T>
MR_PF_DEF Vector<3, T> Cross(const Vector<3, T>& v0, const Vector<3, T>& v1) noexcept
{
    Vector<3, T> result(FMA(v0[1], v1[2], -v0[2] * v1[1]) ,
                        FMA(v0[2], v1[0], -v0[0] * v1[2]) ,
                        FMA(v0[0], v1[1], -v0[1] * v1[0]) );
    return result;
}

template<FloatVectorC T>
MR_PF_DEF bool IsFinite(const T& v) noexcept
{
    bool isFinite = true;
    constexpr unsigned int N = T::Dims;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        isFinite &= IsFinite(v[i]);
    }
    return isFinite;
}

}