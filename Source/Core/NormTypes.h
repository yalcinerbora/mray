#pragma once

#include <concepts>
#include <limits>

static consteval unsigned int ChooseNormAlignment(unsigned int totalSize)
{
    // Do alignment to power of two boundaries;
    // From the classic website
    // https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2

    // Assume Int is at least
    int ts = totalSize;
    ts--;
    ts |= ts >> 1;
    ts |= ts >> 2;
    ts |= ts >> 4;
    ts |= ts >> 8;
    ts++;

    // Set to CUDA data fetch maximum (128 byte)
    return (ts < 32) ? ts : 32;
}

// TODO: Explicitly state the alignments
template <int N, std::unsigned_integral T>
class alignas(ChooseNormAlignment(N * sizeof(T))) UNorm
//class UNorm
{
    public:
    static_assert(N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                  "For UNorm types; N should be 2,4,8,16, or 32");

    MRAY_HYBRID static constexpr T Max();
    MRAY_HYBRID static constexpr T Min();

    private:
    T           v[N];
    public:
    // TODO:
    MRAY_HYBRID constexpr const T&  operator[](int) const;
    MRAY_HYBRID constexpr T&        operator[](int);
};

template <int N, std::signed_integral T>
class alignas(ChooseNormAlignment(N * sizeof(T))) SNorm
{
    public:
    static_assert(N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                  "For UNorm types; N should be 2,4,8,16, or 32");

    MRAY_HYBRID static constexpr T Max();
    MRAY_HYBRID static constexpr T Min();

    private:
    T v[N];
    public:
    MRAY_HYBRID constexpr const T&  operator[](int) const;
    MRAY_HYBRID constexpr T&        operator[](int);
};

template <int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T UNorm<N, T>::Min()
{
    return std::numeric_limits<T>::min();
}

template <int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T UNorm<N, T>::Max()
{
    return std::numeric_limits<T>::max();
}

template <int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& UNorm<N,T>::operator[](int i)
{
    return v[i];
}

// =======================================//

template <int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T SNorm<N, T>::Min()
{
    return std::numeric_limits<T>::min();
}

template <int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T SNorm<N, T>::Max()
{
    return std::numeric_limits<T>::max();
}

template <int N, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& UNorm<N, T>::operator[](int i) const
{
    return v[i];
}

template <int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& SNorm<N, T>::operator[](int i)
{
    return v[i];
}

template <int N, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& SNorm<N, T>::operator[](int i) const
{
    return v[i];
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