#pragma once

#include <concepts>
#include <cassert>
#include <bit>
#include <type_traits>

#include "MathFunctions.h"

namespace Bit
{
    template<std::integral T>
    constexpr T FetchSubPortion(T value, std::array<T, 2> bitRange);

    template<std::unsigned_integral T>
    constexpr T RotateLeft(T value, T shiftAmount);

    template<std::unsigned_integral T>
    constexpr T RotateRight(T value, T shiftAmount);

    template<std::unsigned_integral T>
    constexpr T RequiredBitsToRepresent(T value);

    template<std::unsigned_integral T>
    constexpr T BitReverse(T value, T width = sizeof(T) * CHAR_BIT);

    constexpr uint32_t GenerateFourCC(char byte0, char byte1, char byte2, char byte3);

    namespace NormConversion
    {
        template<std::floating_point R, std::unsigned_integral T>
        MRAY_HYBRID
        constexpr R FromUNorm(T in);

        template<std::unsigned_integral T, std::floating_point R>
        MRAY_HYBRID
        constexpr T ToUNorm(R in);

        template<std::floating_point R, std::signed_integral T>
        MRAY_HYBRID
        constexpr R FromSNorm(T in);

        template<std::signed_integral T, std::floating_point R>
        MRAY_HYBRID
        constexpr T ToSNorm(R in);
    }
}

// Bitset is constexpr in c++23, we should not rely on it on CUDA
// Moreover it "wastes" space in terms of GPU programming
// For GPU case, user may have many bitsets, thus extra space can be an issue
// Allocations should be aligned in this case, not the individual classes
//
// API closely resembles the std::bitset
// TODO: Add shift operations later
template<size_t N>
class Bitset
{
    using Type = std::conditional_t<(N <= 8),   uint8_t,
                 std::conditional_t<(N <= 16),  uint16_t,
                 std::conditional_t<(N <= 32),  uint32_t,
                 std::conditional_t<(N <= 64),  uint64_t,
                 void>>>>;
    static_assert(!std::is_same_v<Type, void>,
                  "MRay bitset at most supports 64-bits!");

    public:
    class BitRef
    {
        friend Bitset<N>;

        private:
        Bitset&     reference;
        uint32_t    index;
        // Constructors & Destructor
        MRAY_HYBRID constexpr           BitRef(Bitset&, uint32_t);

        public:
        MRAY_HYBRID constexpr BitRef&   operator=(bool);
        MRAY_HYBRID constexpr bool      operator~() const;
        MRAY_HYBRID constexpr           operator bool() const;
    };

    private:
    Type bits;

    public:
    // GPU does not have a concept of string so skipping that constructor
    // Constructors & Destructor
                            Bitset() = default;
    MRAY_HYBRID constexpr   Bitset(Type);

    //
    MRAY_HYBRID constexpr bool      operator==(const Bitset& rhs) const;
    MRAY_HYBRID constexpr bool      operator[](uint32_t pos) const;
    MRAY_HYBRID constexpr BitRef    operator[](uint32_t pos);

    MRAY_HYBRID constexpr bool  All() const;
    MRAY_HYBRID constexpr bool  Any() const;
    MRAY_HYBRID constexpr bool  None() const;

    // Specifically utilize 32-bit here
    // 64-bit(std::size_t) may use two registers maybe?
    MRAY_HYBRID constexpr uint32_t  Count() const;
    MRAY_HYBRID constexpr uint32_t  Size() const;

    MRAY_HYBRID constexpr Bitset& operator&=(const Bitset&);
    MRAY_HYBRID constexpr Bitset& operator|=(const Bitset&);
    MRAY_HYBRID constexpr Bitset& operator^=(const Bitset&);
    MRAY_HYBRID constexpr Bitset  operator~() const;

    MRAY_HYBRID constexpr Bitset& Set();
    MRAY_HYBRID constexpr Bitset& Set(uint32_t pos, bool value = true);
    MRAY_HYBRID constexpr Bitset& Reset();
    MRAY_HYBRID constexpr Bitset& Reset(uint32_t pos);
    MRAY_HYBRID constexpr Bitset& Flip();
    MRAY_HYBRID constexpr Bitset& Flip(uint32_t pos);

    MRAY_HYBRID constexpr explicit operator Type();
};

template<size_t N>
MRAY_HYBRID constexpr
Bitset<N> operator&(const Bitset<N>& lhs, const Bitset<N>& rhs);

template<size_t N>
MRAY_HYBRID constexpr
Bitset<N> operator|(const Bitset<N>& lhs, const Bitset<N>& rhs);

template<size_t N>
MRAY_HYBRID constexpr
Bitset<N> operator^(const Bitset<N>& lhs, const Bitset<N>& rhs);

// Bitspan
// Not exact match of a std::span but suits the needs
// and only dynamic extent
template<std::unsigned_integral T>
class Bitspan
{
    private:
    T*                  data;
    uint32_t            size;

    public:
    MRAY_HYBRID constexpr           Bitspan();
    MRAY_HYBRID constexpr           Bitspan(T*, uint32_t bitcount);
    MRAY_HYBRID constexpr           Bitspan(Span<T>);

    MRAY_HYBRID constexpr bool      operator[](uint32_t index) const;
    // Hard to return reference for modification
    MRAY_HYBRID constexpr void      SetBit(uint32_t index, bool) requires (!std::is_const_v<T>);
    MRAY_HYBRID constexpr uint32_t  Size() const;
    MRAY_HYBRID constexpr uint32_t  ByteSize() const;
};

template<std::integral T>
constexpr T Bit::FetchSubPortion(T value, std::array<T, 2> bitRange)
{
    assert(bitRange[0] < bitRange[1]);
    T bitCount = bitRange[1] - bitRange[0];
    T mask = (1 << bitCount) - 1;
    return (value >> bitRange[0]) & mask;
}

template<std::unsigned_integral T>
constexpr T Bit::RotateLeft(T value, T shiftAmount)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    T result = (value << shiftAmount);
    result |= (value >> (Bits - shiftAmount));
    return result;
}

template<std::unsigned_integral T>
constexpr T Bit::RotateRight(T value, T shiftAmount)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    T result = (value >> shiftAmount);
    result |= (value << (Bits - shiftAmount));
    return result;
}

template<std::unsigned_integral T>
constexpr T Bit::RequiredBitsToRepresent(T value)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    return (Bits - T(std::countl_zero(value)));
}

template<std::unsigned_integral T>
constexpr T Bit::BitReverse(T value, T width)
{
    // TODO: Is there a good way to do this than O(n)
    // without lookup table, this may end up in GPU
    T result = 0;
    for(T i = 0; i < width; i++)
    {
        result |= value & 0b1;
        result <<= 1;
        value >>= 1;
    }
    return result;
}

constexpr uint32_t Bit::GenerateFourCC(char byte0, char byte1,
                                       char byte2, char byte3)
{
    constexpr uint32_t p0 = CHAR_BIT * 0;
    constexpr uint32_t p1 = CHAR_BIT * 1;
    constexpr uint32_t p2 = CHAR_BIT * 2;
    constexpr uint32_t p3 = CHAR_BIT * 3;
    auto b0 = static_cast<uint32_t>(byte0);
    auto b1 = static_cast<uint32_t>(byte1);
    auto b2 = static_cast<uint32_t>(byte2);
    auto b3 = static_cast<uint32_t>(byte3);
    return ((b0 << p0) | (b1 << p1) |
            (b2 << p2) | (b3 << p3));
}

template<std::floating_point R, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr R Bit::NormConversion::FromUNorm(T in)
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

template<std::unsigned_integral T, std::floating_point R>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Bit::NormConversion::ToUNorm(R in)
{
    #ifdef MRAY_DEVICE_CODE_PATH
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

template<std::floating_point R, std::signed_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr R Bit::NormConversion::FromSNorm(T in)
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
    // we will use classic 2's complement representation for this conversion
    // so negative numbers will have +1 integer
    //
    // TODO: Check if this is not OK
    // (non-uniform negative/positive may be an issue)
    constexpr R MIN = R(std::numeric_limits<T>::min());
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R NEG_DELTA = R(1) / (R(0) - MIN);
    constexpr R POS_DELTA = R(1) / (MAX - R(0));

    R delta = (in < 0) ? NEG_DELTA : POS_DELTA;
    R result = R(in) * delta;
    return T(result);
}

template<std::signed_integral T, std::floating_point R>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Bit::NormConversion::ToSNorm(R in)
{
    #ifdef MRAY_DEVICE_CODE_PATH
        using std::round;
    #endif

    assert(in >= R(-1) && in <= R(1));
    // Check "FromSNorm" for more info
    //
    constexpr R MIN = R(std::numeric_limits<T>::min());
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R NEG_DIFF = (R(0) - MIN);
    constexpr R POS_DIFF = (MAX - R(0));

    R diff = (in < 0) ? NEG_DIFF : POS_DIFF;
    in *= diff;
    in = round(in);
    return T(in);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>::BitRef::BitRef(Bitset<N>& bs, uint32_t i)
    : reference(bs)
    , index(i)
{}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>::BitRef& Bitset<N>::BitRef::operator=(bool b)
{
    Type mask = (Type(1) << index);
    reference.bits = (b)
            ? reference.bits | mask
            : reference.bits & (~mask);
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitset<N>::BitRef::operator~() const
{
    return static_cast<bool>(((reference.bits >> index) ^ 0x1) & 0x1);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>::BitRef::operator bool() const
{
    return static_cast<bool>((reference.bits >> index) & 0x1);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>::Bitset(Type t)
    : bits(t)
{}

template<size_t N>
MRAY_HYBRID
constexpr bool Bitset<N>::operator==(const Bitset& rhs) const
{
    return (rhs.bits == bits);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitset<N>::operator[](uint32_t pos) const
{
    assert(pos < N);
    return static_cast<bool>((bits >> pos) & 0x1);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>::BitRef Bitset<N>::operator[](uint32_t pos)
{
    assert(pos < N);
    return BitRef(*this, pos);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitset<N>::All() const
{
    return (bits == std::numeric_limits<Type>::max());
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitset<N>::Any() const
{
    return (bits > 0);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitset<N>::None() const
{
    return (bits == 0);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t Bitset<N>::Count() const
{
    return N;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t Bitset<N>::Size() const
{
    return CHAR_BIT * sizeof(Type);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::operator&=(const Bitset& rhs)
{
    bits &= rhs.bits;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::operator|=(const Bitset& rhs)
{
    bits |= rhs.bits;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::operator^=(const Bitset& rhs)
{
    bits ^= rhs.bits;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N> Bitset<N>::operator~() const
{
    return Bitset<N>(~bits);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::Set()
{
    bits = std::numeric_limits<Type>::max();
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::Set(uint32_t pos, bool value)
{
    assert(pos < N);
    BitRef br(*this, pos);
    br = value;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::Reset()
{
    bits = 0x0;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::Reset(uint32_t pos)
{
    assert(pos < N);
    BitRef br(*this, pos);
    br = 0;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::Flip()
{
    bits = ~bits;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>& Bitset<N>::Flip(uint32_t pos)
{
    assert(pos < N);
    BitRef br(*this, pos);
    br = ~br;
    return *this;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N>::operator Type()
{
    return bits;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N> operator&(const Bitset<N>& lhs, const Bitset<N>& rhs)
{
    return lhs.bits & rhs.bits;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N> operator|(const Bitset<N>& lhs, const Bitset<N>& rhs)
{
    return lhs.bits | rhs.bits;
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitset<N> operator^(const Bitset<N>& lhs, const Bitset<N>& rhs)
{
    return lhs.bits ^ rhs.bits;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitspan<T>::Bitspan()
    : data(nullptr)
    , size(0u)
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitspan<T>::Bitspan(T* ptr, uint32_t bitcount)
    : data(ptr)
    , size(bitcount)
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitspan<T>::Bitspan(Span<T> s)
    : data(s.data())
    , size(MathFunctions::NextMultiple(s.size(), sizeof(T)))
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitspan<T>::operator[](uint32_t index) const
{
    assert(index < size && "Out of range access on bitspan!");

    size_t wordIndex = index / sizeof(T);
    size_t wordLocalIndex = index % sizeof(T);

    return data[wordIndex] >> wordLocalIndex;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Bitspan<T>::SetBit(uint32_t index, bool v) requires (!std::is_const_v<T>)
{
    assert(index < size && "Out of range access on bitspan!");

    size_t wordIndex = index / sizeof(T);
    size_t wordLocalIndex = index % sizeof(T);

    T localMask = std::numeric_limits<T>::max();
    localMask &= (static_cast<T>(v) << wordLocalIndex);

    data[wordIndex] &= localMask;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t Bitspan<T>::Size() const
{
    return size;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t Bitspan<T>::ByteSize() const
{
    return MathFunctions::NextMultiple<uint32_t>(size, static_cast<uint32_t>(sizeof(T)));
}