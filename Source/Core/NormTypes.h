#pragma once

#include <concepts>
#include <limits>

#include "MathForward.h"
#include "NormConvFunctions.h"

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
    constexpr           UNorm() = default;
    template<std::convertible_to<T> C>
    MR_PF_DECL explicit UNorm(Span<const C, N> data) noexcept;
    template<std::unsigned_integral... C>
    MR_PF_DECL explicit UNorm(C... vals) noexcept requires(sizeof...(C) == N);
    template<FloatC F>
    MR_PF_DECL explicit UNorm(Vector<N, F>) noexcept;
    // TODO:
    MR_PF_DECL const T& operator[](unsigned int) const noexcept;
    MR_PF_DECL T&       operator[](unsigned int) noexcept;
    MR_PF_DECL static T Max() noexcept;
    MR_PF_DECL static T Min() noexcept;

    MR_PF_DECL const std::array<T, N>&   AsArray() const noexcept;
    MR_PF_DECL std::array<T, N>&         AsArray() noexcept;
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
    std::array<T, N> v;

    public:
    // Constructors & Destructor
    constexpr           SNorm() = default;
    template<std::convertible_to<T> C>
    MR_PF_DECL explicit SNorm(Span<const C, N> data) noexcept;
    template<std::signed_integral... C>
    MR_PF_DECL explicit SNorm(C... vals) noexcept requires(sizeof...(C) == N);
    template<FloatC F>
    MR_PF_DECL explicit SNorm(Vector<N, F>) noexcept;
    // TODO:
    MR_PF_DECL const T& operator[](unsigned int) const noexcept;
    MR_PF_DECL T&       operator[](unsigned int) noexcept;
    MR_PF_DECL static T Max() noexcept;
    MR_PF_DECL static T Min() noexcept;
    //
    MR_PF_DECL const std::array<T, N>&   AsArray() const noexcept;
    MR_PF_DECL std::array<T, N>&         AsArray() noexcept;

};

template <unsigned int N, std::unsigned_integral T>
template<std::convertible_to<T> C>
MR_PF_DEF UNorm<N, T>::UNorm(Span<const C, N> data) noexcept
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, std::unsigned_integral T>
template<FloatC F>
MR_PF_DEF UNorm<N, T>::UNorm(Vector<N, F> data) noexcept
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        using NormConversion::ToUNorm;
        v[i] = ToUNorm<T>(data[i]);
    }
}

template <unsigned int N, std::unsigned_integral T>
template<std::unsigned_integral... C>
MR_PF_DEF UNorm<N, T>::UNorm(C... vals) noexcept
requires(sizeof...(C) == N)
    : v{static_cast<T>(vals)...}
{}

template <unsigned int N, std::unsigned_integral T>
MR_PF_DEF T& UNorm<N,T>::operator[](unsigned int i) noexcept
{
    return v[i];
}

template <unsigned int N, std::unsigned_integral T>
MR_PF_DEF const T& UNorm<N, T>::operator[](unsigned int i) const noexcept
{
    return v[i];
}

template <unsigned int N, std::unsigned_integral T>
MR_PF_DEF T UNorm<N, T>::Min() noexcept
{
    return std::numeric_limits<T>::min();
}

template <unsigned int N, std::unsigned_integral T>
MR_PF_DEF T UNorm<N, T>::Max() noexcept
{
    return std::numeric_limits<T>::max();
}

template <unsigned int N, std::unsigned_integral T>
MR_PF_DEF const std::array<T, N>& UNorm<N, T>::AsArray() const noexcept
{
    return v;
}

template <unsigned int N, std::unsigned_integral T>
MR_PF_DEF std::array<T, N>& UNorm<N, T>::AsArray() noexcept
{
    return v;
}

// =======================================//
template <unsigned int N, std::signed_integral T>
template<std::convertible_to<T> C>
MR_PF_DEF SNorm<N, T>::SNorm(Span<const C, N> data) noexcept
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, std::signed_integral T>
template<FloatC F>
MR_PF_DEF SNorm<N, T>::SNorm(Vector<N, F> data) noexcept
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        using NormConversion::ToSNorm;
        v[i] = ToSNorm<T>(data[i]);
    }
}

template <unsigned int N, std::signed_integral T>
template<std::signed_integral... C>
MR_PF_DEF SNorm<N, T>::SNorm(C... vals) noexcept
requires(sizeof...(C) == N)
    : v{static_cast<T>(vals)...}
{}

template <unsigned int N, std::signed_integral T>
MR_PF_DEF T& SNorm<N, T>::operator[](unsigned int i) noexcept
{
    return v[i];
}

template <unsigned int N, std::signed_integral T>
MR_PF_DEF const T& SNorm<N, T>::operator[](unsigned int i) const noexcept
{
    return v[i];
}

template <unsigned int N, std::signed_integral T>
MR_PF_DEF T SNorm<N, T>::Min() noexcept
{
    return std::numeric_limits<T>::min();
}

template <unsigned int N, std::signed_integral T>
MR_PF_DEF T SNorm<N, T>::Max() noexcept
{
    return std::numeric_limits<T>::max();
}

template <unsigned int N, std::signed_integral T>
MR_PF_DEF const std::array<T, N>& SNorm<N, T>::AsArray() const noexcept
{
    return v;
}

template <unsigned int N, std::signed_integral T>
MR_PF_DEF std::array<T, N>& SNorm<N, T>::AsArray() noexcept
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