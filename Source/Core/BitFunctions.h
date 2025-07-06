#pragma once

#include <concepts>
#include <cassert>
#include <bit>
#include <type_traits>

#include <climits>

#include "Math.h"

namespace Bit
{
    template <class T>
    using TPair = std::array<T, 2>;

    template<std::integral T>
    constexpr T FetchSubPortion(T value, TPair<T> bitRange);

    template<std::integral T, std::integral C>
    requires (std::convertible_to<C, T>)
    [[nodiscard]]
    constexpr T SetSubPortion(T value, C in, TPair<T> bitRange);

    template<size_t... Is, std::unsigned_integral... Ts>
    requires(sizeof...(Is) == sizeof...(Ts))
    constexpr std::tuple_element_t<0, std::tuple<Ts...>>
    Compose(Ts... values);

    template<std::unsigned_integral T>
    constexpr T     RotateLeft(T value, T shiftAmount);

    template<std::unsigned_integral T>
    constexpr T     RotateRight(T value, T shiftAmount);

    // 128-bit variant (used for BC stuff)
    constexpr
    TPair<uint64_t> RotateLeft(TPair<uint64_t> value, uint32_t shiftAmount);

    constexpr
    TPair<uint64_t> RotateRight(TPair<uint64_t> value, uint32_t shiftAmount);

    template<std::unsigned_integral T>
    constexpr T     RequiredBitsToRepresent(T value);

    template<std::unsigned_integral T>
    constexpr T     BitReverse(T value,
                               T width = sizeof(T) * CHAR_BIT);

    template<std::unsigned_integral T>
    constexpr T     CountLZero(T value);

    template<std::unsigned_integral T>
    constexpr T     CountTZero(T value);

    template<std::unsigned_integral T>
    constexpr T     CountLOne(T value);

    template<std::unsigned_integral T>
    constexpr T     CountTOne(T value);

    template<std::unsigned_integral T>
    constexpr T     PopC(T value);

    constexpr
    uint32_t        GenerateFourCC(char byte0, char byte1,
                                   char byte2, char byte3);

    namespace NormConversion
    {
        template<std::floating_point R, std::unsigned_integral T>
        MRAY_HYBRID
        constexpr R FromUNorm(T in);

        template<std::floating_point R, uint32_t Bits,
                 std::unsigned_integral T>
        MRAY_HYBRID
        constexpr R FromUNormVaryingInsane(T value);

        template<std::floating_point R, std::unsigned_integral T>
        MRAY_HYBRID
        constexpr R FromUNormVarying(T in, T bits);

        template<std::unsigned_integral T, std::floating_point R>
        MRAY_HYBRID
        constexpr T ToUNorm(R in);

        template<std::unsigned_integral T, std::floating_point R>
        MRAY_HYBRID
        constexpr T ToUNormVarying(R in, T bits);

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
    static constexpr Type MASK = Type((size_t(1) << N) - 1);

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
    // How many bits is set
    MRAY_HYBRID constexpr uint32_t  PopCount() const;

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

template<std::integral T>
constexpr T Bit::FetchSubPortion(T value, std::array<T, 2> bitRange)
{
    assert(bitRange[0] < bitRange[1]);
    T bitCount = bitRange[1] - bitRange[0];
    T mask = (T(1) << bitCount) - 1;
    return (value >> bitRange[0]) & mask;
}

template<std::integral T, std::integral C>
requires (std::convertible_to<C, T>)
constexpr T Bit::SetSubPortion(T value, C in, std::array<T, 2> bitRange)
{
    assert(bitRange[0] < bitRange[1]);
    T inT = T(in);
    T bitCount = bitRange[1] - bitRange[0];
    T mask0 = (T(1) << bitCount) - 1;
    T mask1 = ~(mask0 << bitRange[0]);
    return (value & mask1) | ((inT & mask0) << bitRange[0]);
}

template<size_t... Is, std::unsigned_integral... Ts>
requires(sizeof...(Is) == sizeof...(Ts))
constexpr std::tuple_element_t<0, std::tuple<Ts...>>
Bit::Compose(Ts... values)
{
    constexpr uint32_t E = sizeof...(Is);
    using RetT = std::tuple_element_t<0, std::tuple<Ts...>>;
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
    UNROLL_LOOP
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
constexpr T Bit::RotateLeft(T value, T shiftAmount)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    assert(shiftAmount < Bits);
    if(shiftAmount == 0) return value;

    T result = (value << shiftAmount);
    result |= (value >> (Bits - shiftAmount));
    return result;
}

template<std::unsigned_integral T>
constexpr T Bit::RotateRight(T value, T shiftAmount)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    assert(shiftAmount < Bits);
    if(shiftAmount == 0) return value;

    T result = (value >> shiftAmount);
    result |= (value << (Bits - shiftAmount));
    return result;
}

constexpr inline
typename Bit::TPair<uint64_t>
Bit::RotateLeft(TPair<uint64_t> value, uint32_t shift)
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

constexpr inline
typename Bit::TPair<uint64_t>
Bit::RotateRight(TPair<uint64_t> value, uint32_t shift)
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
constexpr T Bit::RequiredBitsToRepresent(T value)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    return (Bits - CountLZero(value));
}

template<std::unsigned_integral T>
constexpr T Bit::BitReverse(T value, T width)
{
    constexpr auto Func = [](T value, T width) -> T
    {
        // TODO: Is there a good way to do this
        // other than O(n)
        T result = 0;
        for(T i = 0; i < width; i++)
        {
            result |= value & 0b1;
            result <<= 1;
            value >>= 1;
        }
        return result;
    };

    if(std::is_constant_evaluated())
    {
        return Func(value, width);
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
            return Func(value, width);
        }
        #endif
    }

}

template<std::unsigned_integral T>
constexpr T Bit::CountLZero(T value)
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
constexpr T Bit::CountTZero(T value)
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
constexpr T Bit::CountLOne(T value)
{
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        T vR = ~value;
        return CountLZero(vR);
    #else
        return std::countl_one<T>(value);
    #endif
}

template<std::unsigned_integral T>
constexpr T Bit::CountTOne(T value)
{
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        T vR = ~value;
        return CountTZero(vR);
    #else
        return std::countr_one<T>(value);
    #endif
}

template<std::unsigned_integral T>
constexpr T Bit::PopC(T value)
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

template<std::floating_point R, uint32_t BITS,
         std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr R Bit::NormConversion::FromUNormVaryingInsane(T value)
{
    // We can do divide an all, but we get our hands dirty for
    // bisecting BC compressions for color conversion
    // So lets do a cool version of unorm generation directly
    // embedding to the matissa of the float
     // Directly embed these to mantissa
    static_assert(std::numeric_limits<Float>::is_iec559,
                  "IEEE-754 floats are required for this function");
    using IntT = std::conditional_t<(std::is_same_v<Float, float>),
                                    uint32_t,
                                    uint64_t>;
    constexpr IntT MANTISSA_SIZE = std::is_same_v<Float, float>
                                    ? 23u
                                    : 52u;
    static_assert(BITS <= MANTISSA_SIZE, "Bit count exceeds mantissa");
    assert(value >> BITS == 0);
    // Set the exponent side via the compiler
    // lets not get fancy (thanks to bit_cast)
    IntT rBits = std::bit_cast<IntT>(R(1));
    IntT input = value;
    input <<= (MANTISSA_SIZE - BITS);
    // Here is the fancy part we copy the value to the mantissa
    // starting from MSB of the mantissa
    constexpr IntT StampCount = Math::DivideUp(MANTISSA_SIZE, BITS);
    UNROLL_LOOP
    for(IntT i = 0; i < StampCount; i++)
    {
        rBits |= input;
        input >>= BITS;
    }
    // All this work makes sense when Float is 1.0, extract 1.
    //
    // Is this worth it? Definately no.
    // NVCC recommends to not mix float -> int conversions, we did that
    // a classical approach
    // R(input) / R((1 << bitSize) - 1) also have type conversions
    // so we did not gained anything
    // Furthermore, this is value not compatible with this approach.
    //
    // But no floating point division so its performant even with a loop?
    // porbably not. So do not do this :)
    return R(1) - std::bit_cast<R>(rBits);
}

template<std::floating_point R, std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr R Bit::NormConversion::FromUNormVarying(T in,T bits)
{
    R max = R((T(1) << bits) - T(1));
    R min = R(std::numeric_limits<T>::min());
    R delta = 1 / (max - min);
    // TODO: Specialize using intrinsics maybe?
    // For GPU (GPUs should have these?)
    // Also check more precise way to do this (if available?)
    R result = min + static_cast<R>(in) * delta;
    return result;
}

template<std::unsigned_integral T, std::floating_point R>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Bit::NormConversion::ToUNorm(R in)
{
    #ifndef MRAY_DEVICE_CODE_PATH
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

template<std::unsigned_integral T, std::floating_point R>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Bit::NormConversion::ToUNormVarying(R in, T bits)
{
    #ifndef MRAY_DEVICE_CODE_PATH
        using std::round;
    #endif

    assert(in >= R(0) && in <= R(1));
    R max = R((T(1) << bits) - T(1));
    R min = R(std::numeric_limits<T>::min());
    R diff = (max - min);

    in *= diff;
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
    // we will use DX version since these will be used to
    // fetch BC textures.
    static_assert(std::numeric_limits<T>::max() +
                  std::numeric_limits<T>::min() == -1,
                  "Not two's complement int type?");
    constexpr T MIN = std::numeric_limits<T>::min();
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R DELTA = R(1) / (MAX - R(0));

    in = (in == MIN) ? (in + 1) : in;
    R result = R(in) * DELTA;
    return result;
}

template<std::signed_integral T, std::floating_point R>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Bit::NormConversion::ToSNorm(R in)
{
    #ifndef MRAY_DEVICE_CODE_PATH
        using std::round;
    #endif

    assert(in >= R(-1) && in <= R(1));
    // Check "FromSNorm" for more info
    //
    static_assert(std::numeric_limits<T>::max() +
                  std::numeric_limits<T>::min() == -1,
                  "Not two's complement int type?");
    constexpr R MAX = R(std::numeric_limits<T>::max());
    constexpr R DIFF = (MAX - R(0));

    in *= DIFF;
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
    Type mask = Type(1u << index);
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
    return (bits == MASK);
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
constexpr uint32_t Bitset<N>::PopCount() const
{
    return Bit::PopC<Type>(MASK & bits);
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
    bits = MASK;
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