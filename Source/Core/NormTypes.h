#pragma once

#include <concepts>
#include <limits>

#include "MathForward.h"
#include "BitFunctions.h"

static consteval unsigned int ChooseNormAlignment(unsigned int totalSize)
{
    // Do alignment to power of two boundaries;
    // From the classic website
    // https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2

    // Assume Int is at least
    unsigned int ts = totalSize;
    ts--;
    ts |= ts >> 1;
    ts |= ts >> 2;
    ts |= ts >> 4;
    ts |= ts >> 8;
    ts++;

    // Set to CUDA data fetch maximum (128 byte)
    return (ts < 32u) ? ts : 32u;
}

// TODO: Explicitly state the alignments
template <unsigned int N, std::unsigned_integral T>
class alignas(ChooseNormAlignment(N * sizeof(T))) UNorm
{
    static_assert(N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                  "For UNorm types; N should be 2,4,8,16, or 32");

    public:
    using InnerType                     = T;
    static constexpr unsigned int Dims  = N;
    private:
    std::array<T, N>                v;

    public:
    // Constructors & Destructor
    constexpr                       UNorm() = default;
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  UNorm(Span<const C, N> data);
    template<std::floating_point F>
    MRAY_HYBRID constexpr explicit  UNorm(Vector<N, F>);
    // TODO:
    MRAY_HYBRID constexpr const T&  operator[](unsigned int) const;
    MRAY_HYBRID constexpr T&        operator[](unsigned int);
    MRAY_HYBRID static constexpr T  Max();
    MRAY_HYBRID static constexpr T  Min();

    MRAY_HYBRID
    constexpr const std::array<T, N>&   AsArray() const;
    MRAY_HYBRID
    constexpr std::array<T, N>&         AsArray();
};

template <unsigned int N, std::signed_integral T>
class alignas(ChooseNormAlignment(N * sizeof(T))) SNorm
{
    static_assert(N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                  "For UNorm types; N should be 2,4,8,16, or 32");
    public:
    using InnerType                     = T;
    static constexpr unsigned int Dims  = N;

    private:
    std::array<T, N>                v;

    public:
    // Constructors & Destructor
    constexpr                       SNorm() = default;
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  SNorm(Span<const C, N> data);
    template<std::floating_point F>
    MRAY_HYBRID constexpr explicit  SNorm(Vector<N, F>);
    // TODO:
    MRAY_HYBRID constexpr const T&  operator[](unsigned int) const;
    MRAY_HYBRID constexpr T&        operator[](unsigned int);
    MRAY_HYBRID static constexpr T  Max();
    MRAY_HYBRID static constexpr T  Min();
    MRAY_HYBRID
    constexpr const std::array<T, N>&   AsArray() const;
    MRAY_HYBRID
    constexpr std::array<T, N>&         AsArray();

};

template <unsigned int N, std::unsigned_integral T>
template<std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr UNorm<N, T>::UNorm(Span<const C, N> data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, std::unsigned_integral T>
template<std::floating_point F>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr UNorm<N, T>::UNorm(Vector<N, F> data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        using Bit::NormConversion::ToUNorm;
        v[i] = ToUNorm<T>(data[i]);
    }
}

template <unsigned int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& UNorm<N,T>::operator[](unsigned int i)
{
    return v[i];
}

template <unsigned int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& UNorm<N, T>::operator[](unsigned int i) const
{
    return v[i];
}

template <unsigned int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T UNorm<N, T>::Min()
{
    return std::numeric_limits<T>::min();
}

template <unsigned int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T UNorm<N, T>::Max()
{
    return std::numeric_limits<T>::max();
}

template <unsigned int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const std::array<T, N>& UNorm<N, T>::AsArray() const
{
    return v;
}

template <unsigned int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr std::array<T, N>& UNorm<N, T>::AsArray()
{
    return v;
}

// =======================================//
template <unsigned int N, std::signed_integral T>
template<std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr SNorm<N, T>::SNorm(Span<const C, N> data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, std::signed_integral T>
template<std::floating_point F>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr SNorm<N, T>::SNorm(Vector<N, F> data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        using Bit::NormConversion::ToSNorm;
        v[i] = ToSNorm<T>(data[i]);
    }
}

template <unsigned int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& SNorm<N, T>::operator[](unsigned int i)
{
    return v[i];
}

template <unsigned int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& SNorm<N, T>::operator[](unsigned int i) const
{
    return v[i];
}

template <unsigned int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T SNorm<N, T>::Min()
{
    return std::numeric_limits<T>::min();
}

template <unsigned int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T SNorm<N, T>::Max()
{
    return std::numeric_limits<T>::max();
}

template <unsigned int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const std::array<T, N>& SNorm<N, T>::AsArray() const
{
    return v;
}

template <unsigned int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr std::array<T, N>& SNorm<N, T>::AsArray()
{
    return v;
}

// Word Types
using UNorm4x8      = UNorm<4, uint8_t>;
using UNorm2x16     = UNorm<2, uint16_t>;
using SNorm4x8      = SNorm<4, int8_t>;
using SNorm2x16     = SNorm<2, int16_t>;

// Double-word Types
using UNorm8x8      = UNorm<8, uint8_t>;
using UNorm4x16     = UNorm<4, uint16_t>;
using UNorm2x32     = UNorm<2, uint32_t>;
using SNorm8x8      = SNorm<8, int8_t>;
using SNorm4x16     = SNorm<4, int16_t>;
using SNorm2x32     = SNorm<2, int32_t>;

// Quad-word Types
using UNorm16x8     = UNorm<16, uint8_t>;
using UNorm8x16     = UNorm<8, uint16_t>;
using UNorm4x32     = UNorm<4, uint32_t>;
using UNorm2x64     = UNorm<2, uint64_t>;
using SNorm16x8     = SNorm<16, int8_t>;
using SNorm8x16     = SNorm<8, int16_t>;
using SNorm4x32     = SNorm<4, int32_t>;
using SNorm2x64     = SNorm<2, int64_t>;

// 8x-word Types
using UNorm32x8     = UNorm<32, uint8_t>;
using UNorm16x16    = UNorm<16, uint16_t>;
using UNorm8x32     = UNorm<8, uint32_t>;
using UNorm4x64     = UNorm<4, uint64_t>;
using SNorm32x8     = SNorm<32, int8_t>;
using SNorm16x16    = SNorm<16, int16_t>;
using SNorm8x32     = SNorm<8, int32_t>;
using SNorm4x64     = SNorm<4, int64_t>;

static_assert(ArrayLikeC<UNorm4x8>, "UNorm4x8 is not array like!");
static_assert(ArrayLikeC<SNorm4x8>, "SNorm4x8 is not array like!");

static_assert(ImplicitLifetimeC<UNorm4x8>, "UNorm4x8 is not implicit lifetime class");
static_assert(ImplicitLifetimeC<SNorm4x8>, "SNorm4x8 is not implicit lifetime class");