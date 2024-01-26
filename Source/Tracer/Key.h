#pragma once

#include <cstdint>
#include <iomanip>

#include <fmt/format.h>
#include <fmt/core.h>

#include "Core/Definitions.h"

// Main Key Logic, All Primitive, Transform, Material, Medium,
// types will have these accessing bits
// It has two portions, "Batch" portion which isolates the type,
// and index portion which isolates index of that batch
template <std::unsigned_integral T, uint32_t BBits, uint32_t IBits>
class alignas(sizeof(T)) KeyT
{
    public:
    using Type = T;

    // Constants
    static constexpr T          BatchBits = BBits;
    static constexpr T          IdBits = IBits;

    static constexpr T          IdMask = (0x1ull << IdBits) - 1;
    static constexpr T          BatchMask = ((0x1ull << BatchBits) - 1) << IdBits;

    private:
    // Props
    T                           value;

    public:
    // Constructors & Destructor
                                KeyT() = default;
    MRAY_HYBRID constexpr       KeyT(T v);
    //
    MRAY_HYBRID constexpr       operator T() const;
    MRAY_HYBRID constexpr       operator T&();
    // Access
    MRAY_HYBRID constexpr T     FetchBatchPortion() const;
    MRAY_HYBRID constexpr T     FetchIndexPortion() const;

    MRAY_HYBRID
    static constexpr KeyT       CombinedKey(T batch, T id);
    MRAY_HYBRID
    static constexpr KeyT       InvalidKey();

    // Sanity Checks
    static_assert((IdBits + BatchBits) == std::numeric_limits<T>::digits,
                  "Bits representing portions of HitKey should complement each other.");
    static_assert((IdMask | BatchMask) == std::numeric_limits<T>::max() &&
                  (IdMask & BatchMask) == std::numeric_limits<T>::min(),
                  "Masks representing portions of HitKey should complement each other.");
};

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr KeyT<T, BB, IB>::KeyT(T v)
    : value(v)
{}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr KeyT<T, BB, IB>::operator T() const
{
    return value;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr KeyT<T, BB, IB>::operator T&()
{
    return value;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T KeyT<T, BB, IB>::FetchBatchPortion() const
{
    return (value & BatchMask) >> IdBits;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T KeyT<T, BB, IB>::FetchIndexPortion() const
{
    return value & IdMask;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr KeyT<T, BB, IB> KeyT<T, BB, IB>::CombinedKey(T batch, T id)
{
    return KeyT((batch << IdBits) | (id & IdMask));
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr KeyT<T, BB, IB> KeyT<T, BB, IB>::InvalidKey()
{
    return KeyT(std::numeric_limits<T>::max());
}

// Format helpers for debugging
template<std::unsigned_integral T, uint32_t BB, uint32_t IB>
auto format_as(const KeyT<T, BB, IB>& k)
{
    std::string s = fmt::format("[{:0>{}b}|{:0>{}b}]",
                                k.FetchBatchPortion(), BB,
                                k.FetchIndexPortion(), IB);
    return s;
}


// Rely on overload resolution to write in hexedecimal
template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
struct HexKeyT : public KeyT<T, BB, IB> {};

template<std::unsigned_integral T, uint32_t BB, uint32_t IB>
auto format_as(const HexKeyT<T, BB, IB>& k)
{
    std::string s = fmt::format("[{:0>{}X}|{:0>{}X}]",
                                k.FetchBatchPortion(), (BB + 3) / 4,
                                k.FetchIndexPortion(), (IB + 3) / 4);
    return s;
}