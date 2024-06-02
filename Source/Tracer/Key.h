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
    static constexpr T  BatchBits   = BBits;
    static constexpr T  IdBits      = IBits;

    static constexpr T  BatchMask   = ((0x1ull << BatchBits) - 1) << IdBits;
    static constexpr T  IdMask      = ((0x1ull << IdBits)    - 1) << 0;

    private:
    // Props
    T                   value;

    public:
    // Constructors & Destructor
                                    KeyT() = default;
    MRAY_HYBRID explicit constexpr  KeyT(T v);
    //
    MRAY_HYBRID explicit constexpr  operator T() const;
    MRAY_HYBRID explicit constexpr  operator T&();

    MRAY_HYBRID bool                operator==(const KeyT& rhs) const;

    // Access
    MRAY_HYBRID constexpr T         FetchBatchPortion() const;
    MRAY_HYBRID constexpr T         FetchIndexPortion() const;

    MRAY_HYBRID
    static constexpr KeyT           CombinedKey(T batch, T id);
    MRAY_HYBRID
    static constexpr KeyT           InvalidKey();

    // Sanity Checks
    static_assert((IdBits + BatchBits) == std::numeric_limits<T>::digits,
                  "Bits representing portions of HitKey should complement each other.");
    static_assert((IdMask | BatchMask) == std::numeric_limits<T>::max() &&
                  (IdMask & BatchMask) == std::numeric_limits<T>::min(),
                  "Masks representing portions of HitKey should complement each other.");
};

// Triple key type, used for surface work keys,
template <std::unsigned_integral T, uint32_t FBits, uint32_t BBits, uint32_t IBits>
class alignas(sizeof(T)) TriKeyT
{
    public:
    using Type = T;

    // Constants
    static constexpr T  FlagBits    = FBits;
    static constexpr T  BatchBits   = BBits;
    static constexpr T  IdBits      = IBits;

    static constexpr T  FlagMask    = ((0x1ull << FlagBits)  - 1) << (BatchBits + IdBits);
    static constexpr T  BatchMask   = ((0x1ull << BatchBits) - 1) << (IdBits);
    static constexpr T  IdMask      = ((0x1ull << IdBits)    - 1) << 0;

    private:
    // Props
    T                               value;

    public:
    // Constructors & Destructor
                                    TriKeyT() = default;
    MRAY_HYBRID explicit constexpr  TriKeyT(T v);
    //
    MRAY_HYBRID explicit constexpr  operator T() const;
    MRAY_HYBRID explicit constexpr  operator T&();

    MRAY_HYBRID bool                operator==(const TriKeyT& rhs) const;

    // Access
    MRAY_HYBRID constexpr T         FetchFlagPortion() const;
    MRAY_HYBRID constexpr T         FetchBatchPortion() const;
    MRAY_HYBRID constexpr T         FetchIndexPortion() const;

    MRAY_HYBRID
    static constexpr TriKeyT        CombinedKey(T flag, T batch, T id);
    MRAY_HYBRID
    static constexpr TriKeyT        InvalidKey();

    // Sanity Checks
    static_assert((IdBits + BatchBits + FlagBits) == std::numeric_limits<T>::digits,
                  "Bits representing portions of HitKey should complement each other.");
    static_assert((IdMask | BatchMask | FlagMask) == std::numeric_limits<T>::max() &&
                  (IdMask & BatchMask & FlagMask) == std::numeric_limits<T>::min(),
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
bool KeyT<T, BB, IB>::operator==(const KeyT& rhs) const
{
    return (value == rhs.value);
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
    assert(batch <= BatchMask);
    assert(id <= IdMask);
    return KeyT((batch << IdBits) | (id & IdMask));
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr KeyT<T, BB, IB> KeyT<T, BB, IB>::InvalidKey()
{
    return KeyT(std::numeric_limits<T>::max());
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr TriKeyT<T, FB, BB, IB>::TriKeyT(T v)
    : value(v)
{}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr TriKeyT<T, FB, BB, IB>::operator T() const
{
    return value;
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr TriKeyT<T, FB, BB, IB>::operator T&()
{
    return value;
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
bool TriKeyT<T, FB, BB, IB>::operator==(const TriKeyT& rhs) const
{
    return (value == rhs.value);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T TriKeyT<T, FB, BB, IB>::FetchFlagPortion() const
{
    return (value & FlagMask) >> (BatchBits + IdBits);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T TriKeyT<T, FB, BB, IB>::FetchBatchPortion() const
{
    return (value & BatchMask) >> IdBits;
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T TriKeyT<T, FB, BB, IB>::FetchIndexPortion() const
{
    return (value & IdMask);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr TriKeyT<T, FB, BB, IB> TriKeyT<T, FB, BB, IB>::CombinedKey(T flag, T batch, T id)
{
    assert(flag <= FlagMask);
    assert(batch <= BatchMask);
    assert(id <= IdMask);
    T flagPortion   = flag << (IdBits + BatchBits);
    T batchPortion  = batch << IdBits;
    T idPortion     = id;
    return TriKeyT(flagPortion | batchPortion | idPortion);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr TriKeyT<T, FB, BB, IB> TriKeyT<T, FB, BB, IB>::InvalidKey()
{
    return TriKeyT(std::numeric_limits<T>::max());
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

template<std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
auto format_as(const TriKeyT<T, FB, BB, IB>& k)
{
    std::string s = fmt::format("[{:0>{}b}|{:0>{}b}|{:0>{}b}]",
                                k.FetchFlagPortion(), FB,
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