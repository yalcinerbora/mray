#pragma once

#include <concepts>
#include <cassert>
#include <bit>
#include <type_traits>
#include <climits>

#include "Definitions.h"
#include "Tuple.h"

namespace Bit
{
    template <class T>
    using TPair = std::array<T, 2>;

    template<class To, class From>
    MR_PF_DECL To BitCast(const From& value) noexcept;

    template<std::integral T>
    MR_PF_DECL T FetchSubPortion(T value, TPair<T> bitRange) noexcept;

    template<std::integral T, std::integral C>
    requires (std::convertible_to<C, T>)
    MR_PF_DECL T SetSubPortion(T value, C in, TPair<T> bitRange) noexcept;

    template<size_t... Is, std::unsigned_integral... Ts>
    requires(sizeof...(Is) == sizeof...(Ts))
    MR_PF_DECL TupleElement<0, Tuple<Ts...>>
    Compose(Ts... values) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T RotateLeft(T value, T shiftAmount) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T RotateRight(T value, T shiftAmount) noexcept;

    // 128-bit variant (used for BC stuff)
    MR_PF_DECL TPair<uint64_t> RotateLeft(TPair<uint64_t> value, uint32_t shiftAmount) noexcept;
    MR_PF_DECL TPair<uint64_t> RotateRight(TPair<uint64_t> value, uint32_t shiftAmount) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T RequiredBitsToRepresent(T value) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T BitReverse(T value, T width = sizeof(T) * CHAR_BIT) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T CountLZero(T value) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T CountTZero(T value) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T CountLOne(T value) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T CountTOne(T value) noexcept;

    template<std::unsigned_integral T>
    MR_PF_DECL T PopC(T value) noexcept;

    MR_PF_DECL uint32_t GenerateFourCC(char byte0, char byte1,
                                       char byte2, char byte3) noexcept;
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
    static constexpr Type MASK = Type((size_t(1) << N) - 1);

    public:
    class BitRef
    {
        friend Bitset<N>;

        private:
        Bitset&     reference;
        uint32_t    index;
        // Constructors & Destructor
        MR_HF_DECL          BitRef(Bitset&, uint32_t) noexcept;
        public:
        MR_HF_DECL BitRef&  operator=(bool) noexcept;
        MR_HF_DECL bool     operator~() const noexcept;
        MR_HF_DECL          operator bool() const noexcept;
    };

    private:
    Type bits;

    public:
    // GPU does not have a concept of string so skipping that constructor
    // Constructors & Destructor
                Bitset() = default;
    MR_HF_DECL  Bitset(Type) noexcept;
    //
    MR_HF_DECL bool     operator==(const Bitset& rhs) const noexcept;
    MR_HF_DECL bool     operator[](uint32_t pos) const noexcept;
    MR_HF_DECL BitRef   operator[](uint32_t pos) noexcept;

    MR_HF_DECL bool  All() const noexcept;
    MR_HF_DECL bool  Any() const noexcept;
    MR_HF_DECL bool  None() const noexcept;

    // Specifically utilize 32-bit here
    // 64-bit(std::size_t) may use two registers maybe?
    MR_HF_DECL uint32_t  Count() const noexcept;
    MR_HF_DECL uint32_t  Size() const noexcept;
    // How many bits is set
    MR_HF_DECL uint32_t  PopCount() const noexcept;

    MR_HF_DECL Bitset& operator&=(const Bitset&) noexcept;
    MR_HF_DECL Bitset& operator|=(const Bitset&) noexcept;
    MR_HF_DECL Bitset& operator^=(const Bitset&) noexcept;
    MR_HF_DECL Bitset  operator~() const noexcept;

    MR_HF_DECL Bitset& Set() noexcept;
    MR_HF_DECL Bitset& Set(uint32_t pos, bool value = true) noexcept;
    MR_HF_DECL Bitset& Reset() noexcept;
    MR_HF_DECL Bitset& Reset(uint32_t pos) noexcept;
    MR_HF_DECL Bitset& Flip() noexcept;
    MR_HF_DECL Bitset& Flip(uint32_t pos) noexcept;

    MR_HF_DECL explicit operator Type() noexcept;
};

template<size_t N>
MR_HF_DECL Bitset<N> operator&(const Bitset<N>& lhs, const Bitset<N>& rhs) noexcept;

template<size_t N>
MR_HF_DECL Bitset<N> operator|(const Bitset<N>& lhs, const Bitset<N>& rhs) noexcept;

template<size_t N>
MR_HF_DECL Bitset<N> operator^(const Bitset<N>& lhs, const Bitset<N>& rhs) noexcept;

template<class R, class T>
MR_PF_DEF R Bit::BitCast(const T& value) noexcept
{
    if(std::is_constant_evaluated())
        return std::bit_cast<R>(value);
    // This was failed once on CUDA (OptixIR compilation)
    // So wrapping it around
    #ifndef MRAY_GPU_CODE_PATH
        return std::bit_cast<R>(value);
    #else
        template<class A, class B>
        static constexpr TypeEq = std::is_same_v<R, A> && std::is_same_v<T, B>)

        if constexpr(TypeEq<float, uint32_t>) return __uint_as_float(value);
        if constexpr(TypeEq<uint32_t, float>) return __float_as_uint(value);
        if constexpr(TypeEq<float, int32_t>)  return __int_as_float(value);
        if constexpr(TypeEq<int32_t, float>)  return __float_as_int(value);
        //
        if constexpr(TypeEq<double, uint64_t>)
        {
            int64_t r;
            memcpy(&r, &v, sizeof(int64_t));
            return __longlong_as_double(T(r));
        }
        if constexpr(TypeEq<uint64_t, double>)
        {
            int64_t v = __double_as_longlong(value);
            uint64_t r;
            memcpy(&r, &v, sizeof(uint64_t));
            return r;
        }
        if constexpr(TypeEq<double, int64_t>)  return __longlong_as_double(value);
        if constexpr(TypeEq<int64_t, double>)  return __double_as_longlong(value);
        // Rely on c++ bit_cast for the rest
        return std::bit_cast<R>(value);
    #endif
}

template<std::integral T>
MR_PF_DEF T Bit::FetchSubPortion(T value, std::array<T, 2> bitRange) noexcept
{
    T bitCount = bitRange[1] - bitRange[0];
    assert(bitRange[0] < bitRange[1]);
    assert(bitCount < sizeof(T) * CHAR_BIT);
    if constexpr(std::is_signed_v<T>)
        assert(bitCount >= 0);

    T mask = (T(1) << bitCount) - 1;
    return (value >> bitRange[0]) & mask;
}

template<std::integral T, std::integral C>
requires (std::convertible_to<C, T>)
MR_PF_DEF T Bit::SetSubPortion(T value, C in, std::array<T, 2> bitRange) noexcept
{
    T bitCount = bitRange[1] - bitRange[0];
    assert(bitRange[0] < bitRange[1]);
    assert(bitCount < sizeof(T) * CHAR_BIT);
    if constexpr(std::is_signed_v<T>)
        assert(bitCount >= 0);

    T inT = T(in);
    T mask0 = (T(1) << bitCount) - 1;
    T mask1 = ~(mask0 << bitRange[0]);
    return (value & mask1) | ((inT & mask0) << bitRange[0]);
}

template<size_t... Is, std::unsigned_integral... Ts>
requires(sizeof...(Is) == sizeof...(Ts))
MR_PF_DEF TupleElement<0, Tuple<Ts...>>
Bit::Compose(Ts... values) noexcept
{
    constexpr uint32_t E = sizeof...(Is);
    using RetT = TupleElement<0, Tuple<Ts...>>;
    // Compile time check the bits
    static_assert((Is + ...) <= sizeof(RetT) * CHAR_BIT,
                  "Template \"Is\" must not exceed the entire bit range");
    // Convert the index sequence to runtime
    // TODO: Can we "scan" in comptime via expansion?
    std::array<RetT, E + 1> offsets = {0, Is...};
    // Check the data validity
    if constexpr(MRAY_IS_DEBUG)
    {
        [[maybe_unused]]
        uint32_t i = 1;
        assert
        ((
            (RetT(values) < (RetT(1) << offsets[i++])) &&
            ...
        ));
    }
    // Do scan
    MRAY_UNROLL_LOOP
    for(uint32_t i = 1; i < E; i++)
    {
        offsets[i] += offsets[i - 1];
    }

    uint32_t i = 0;
    RetT result = 0;
    (
        ((void)(result |= (RetT(values) << offsets[i++]))),
        ...
    );
    return result;
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::RotateLeft(T value, T shiftAmount) noexcept
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    assert(shiftAmount < Bits);
    if(shiftAmount == 0) return value;

    T result = (value << shiftAmount);
    result |= (value >> (Bits - shiftAmount));
    return result;
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::RotateRight(T value, T shiftAmount) noexcept
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    assert(shiftAmount < Bits);
    if(shiftAmount == 0) return value;

    T result = (value >> shiftAmount);
    result |= (value << (Bits - shiftAmount));
    return result;
}

MR_PF_DEF typename Bit::TPair<uint64_t>
Bit::RotateLeft(TPair<uint64_t> value, uint32_t shift) noexcept
{
    uint64_t shift64 = uint64_t(shift);
    constexpr uint64_t Bits = sizeof(uint64_t) * CHAR_BIT;
    assert(shift64 < Bits * 2);
    if(shift >= Bits)
    {
        shift64 -= Bits;
        std::swap(value[0], value[1]);
    }
    if(shift64 == 0) return value;

    uint64_t invShift = Bits - shift64;
    TPair<uint64_t> result;
    result[1] = (value[1] << shift64);
    result[1] |= (value[0] >> invShift);
    result[0] = (value[0] << shift64);
    result[0] |= (value[1] >> invShift);
    return result;
}

MR_PF_DEF typename Bit::TPair<uint64_t>
Bit::RotateRight(TPair<uint64_t> value, uint32_t shift) noexcept
{
    uint64_t shift64 = uint64_t(shift);
    constexpr uint64_t Bits = sizeof(uint64_t) * CHAR_BIT;
    assert(shift64 < Bits * 2);
    if(shift64 >= Bits)
    {
        shift64 -= Bits;
        std::swap(value[0], value[1]);
    }
    if(shift == 0) return value;

    uint64_t invShift = Bits - shift64;
    TPair<uint64_t> result;
    result[0] = (value[0] >> shift64);
    result[0] |= (value[1] << invShift);
    result[1] = (value[1] >> shift64);
    result[1] |= (value[0] << invShift);
    return result;
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::RequiredBitsToRepresent(T value) noexcept
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    return (Bits - CountLZero(value));
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::BitReverse(T value, T width) noexcept
{
    constexpr auto Eval = [](T value, T width) -> T
    {
        // TODO: Is there a good way to do this
        // other than O(n)
        T result = T(0);
        for(T i = 0; i < width; i++)
        {
            result <<= T(1);
            result |= value & 0b1;
            value >>= T(1);
        }
        return result;
    };

    if(std::is_constant_evaluated())
    {
        return Eval(value, width);
    }
    else
    {
        #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        {
            if constexpr(std::is_same_v<T, uint64_t>)
                return T(__brevll(value));
            else
                return T(__brev(uint32_t(value)));
        }
        #else
        {
            return Eval(value, width);
        }
        #endif
    }

}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::CountLZero(T value) noexcept
{
    if(std::is_constant_evaluated())
    {
        return static_cast<T>(std::countl_zero<T>(value));
    }
    else
    {
        #ifdef MRAY_DEVICE_CODE_PATH_CUDA
            if constexpr(std::is_same_v<T, uint64_t>)
                return T(__clzll(std::bit_cast<long long int>(value)));
            else if constexpr(std::is_same_v<T, uint32_t>)
                return std::bit_cast<T>(__clz(std::bit_cast<int>(value)));
            else
                return T(__clz(int(value)));
        #else
            return static_cast<T>(std::countl_zero<T>(value));
        #endif
    };
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::CountTZero(T value) noexcept
{
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        // This is intrinsic so its fine
        T vR = BitReverse(value);
        return CountLZero(vR);
    #else
        return T(std::countr_zero<T>(value));
    #endif
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::CountLOne(T value) noexcept
{
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        T vR = ~value;
        return CountLZero(vR);
    #else
        return std::countl_one<T>(value);
    #endif
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::CountTOne(T value) noexcept
{
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        T vR = ~value;
        return CountTZero(vR);
    #else
        return std::countr_one<T>(value);
    #endif
}

template<std::unsigned_integral T>
MR_PF_DEF T Bit::PopC(T value) noexcept
{
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        if constexpr(std::is_same_v<T, uint64_t>)
            return T(__popcll(value));
        else
            return T(__popc(uint32_t(value)));
    #else
        return T(std::popcount<T>(value));
    #endif
}

MR_PF_DEF uint32_t Bit::GenerateFourCC(char byte0, char byte1,
                                       char byte2, char byte3) noexcept
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

template<size_t N>
MR_HF_DECL Bitset<N>::BitRef::BitRef(Bitset<N>& bs, uint32_t i) noexcept
    : reference(bs)
    , index(i)
{}

template<size_t N>
MR_HF_DECL Bitset<N>::BitRef& Bitset<N>::BitRef::operator=(bool b) noexcept
{
    Type mask = Type(1u << index);
    reference.bits = (b)
            ? reference.bits | mask
            : reference.bits & (~mask);
    return *this;
}

template<size_t N>
MR_HF_DECL bool Bitset<N>::BitRef::operator~() const noexcept
{
    return static_cast<bool>(((reference.bits >> index) ^ 0x1) & 0x1);
}

template<size_t N>
MR_HF_DECL Bitset<N>::BitRef::operator bool() const noexcept
{
    return static_cast<bool>((reference.bits >> index) & 0x1);
}

template<size_t N>
MR_HF_DECL Bitset<N>::Bitset(Type t) noexcept
    : bits(t)
{}

template<size_t N>
MR_HF_DECL bool Bitset<N>::operator==(const Bitset& rhs) const noexcept
{
    return (rhs.bits == bits);
}

template<size_t N>
MR_HF_DECL bool Bitset<N>::operator[](uint32_t pos) const noexcept
{
    assert(pos < N);
    return static_cast<bool>((bits >> pos) & 0x1);
}

template<size_t N>
MR_HF_DECL Bitset<N>::BitRef Bitset<N>::operator[](uint32_t pos) noexcept
{
    assert(pos < N);
    return BitRef(*this, pos);
}

template<size_t N>
MR_HF_DECL bool Bitset<N>::All() const noexcept
{
    return (bits == MASK);
}

template<size_t N>
MR_HF_DECL bool Bitset<N>::Any() const noexcept
{
    return (bits > 0);
}

template<size_t N>
MR_HF_DECL bool Bitset<N>::None() const noexcept
{
    return (bits == 0);
}

template<size_t N>
MR_HF_DECL uint32_t Bitset<N>::Count() const noexcept
{
    return N;
}

template<size_t N>
MR_HF_DECL uint32_t Bitset<N>::Size() const noexcept
{
    return CHAR_BIT * sizeof(Type);
}

template<size_t N>
MR_HF_DECL uint32_t Bitset<N>::PopCount() const noexcept
{
    return Bit::PopC<Type>(MASK & bits);
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::operator&=(const Bitset& rhs) noexcept
{
    bits &= rhs.bits;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::operator|=(const Bitset& rhs) noexcept
{
    bits |= rhs.bits;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::operator^=(const Bitset& rhs) noexcept
{
    bits ^= rhs.bits;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N> Bitset<N>::operator~() const noexcept
{
    return Bitset<N>(~bits);
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::Set() noexcept
{
    bits = MASK;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::Set(uint32_t pos, bool value) noexcept
{
    assert(pos < N);
    BitRef br(*this, pos);
    br = value;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::Reset() noexcept
{
    bits = 0x0;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::Reset(uint32_t pos) noexcept
{
    assert(pos < N);
    BitRef br(*this, pos);
    br = 0;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::Flip() noexcept
{
    bits = ~bits;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>& Bitset<N>::Flip(uint32_t pos) noexcept
{
    assert(pos < N);
    BitRef br(*this, pos);
    br = ~br;
    return *this;
}

template<size_t N>
MR_HF_DECL Bitset<N>::operator Type() noexcept
{
    return bits;
}

template<size_t N>
MR_HF_DECL Bitset<N> operator&(const Bitset<N>& lhs, const Bitset<N>& rhs) noexcept
{
    return lhs.bits & rhs.bits;
}

template<size_t N>
MR_HF_DECL Bitset<N> operator|(const Bitset<N>& lhs, const Bitset<N>& rhs) noexcept
{
    return lhs.bits | rhs.bits;
}

template<size_t N>
MR_HF_DECL Bitset<N> operator^(const Bitset<N>& lhs, const Bitset<N>& rhs) noexcept
{
    return lhs.bits ^ rhs.bits;
}