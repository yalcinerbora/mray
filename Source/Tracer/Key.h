#pragma once

#include <cstdint>

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
    constexpr           KeyT() = default;
    MR_PF_DECL explicit KeyT(T v) noexcept;
    //
    MR_PF_DECL explicit operator T() const noexcept;
    MR_PF_DECL explicit operator T() noexcept;

    MR_PF_DECL bool     operator<(const KeyT& rhs) const noexcept;
    MR_PF_DECL bool     operator==(const KeyT& rhs) const noexcept;

    // Access
    MR_PF_DECL T        FetchBatchPortion() const noexcept;
    MR_PF_DECL T        FetchIndexPortion() const noexcept;

    MR_PF_DECL static KeyT  CombinedKey(T batch, T id) noexcept;
    MR_PF_DECL static KeyT  InvalidKey() noexcept;

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
    constexpr           TriKeyT() = default;
    MR_PF_DECL explicit TriKeyT(T v) noexcept;
    //
    MR_PF_DECL explicit operator T() const noexcept;
    MR_PF_DECL explicit operator T() noexcept;

    MR_PF_DECL bool     operator==(const TriKeyT& rhs) const noexcept;
    MR_PF_DECL bool     operator<(const TriKeyT& rhs) const noexcept;

    // Access
    MR_PF_DECL T         FetchFlagPortion() const noexcept;
    MR_PF_DECL T         FetchBatchPortion() const noexcept;
    MR_PF_DECL T         FetchIndexPortion() const noexcept;

    MR_PF_DECL static TriKeyT   CombinedKey(T flag, T batch, T id) noexcept;
    MR_PF_DECL static TriKeyT   InvalidKey() noexcept;

    // Sanity Checks
    static_assert((IdBits + BatchBits + FlagBits) == std::numeric_limits<T>::digits,
                  "Bits representing portions of HitKey should complement each other.");
    static_assert((IdMask | BatchMask | FlagMask) == std::numeric_limits<T>::max() &&
                  (IdMask & BatchMask & FlagMask) == std::numeric_limits<T>::min(),
                  "Masks representing portions of HitKey should complement each other.");
};

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
KeyT<T, BB, IB>::KeyT(T v) noexcept
    : value(v)
{}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
KeyT<T, BB, IB>::operator T() const noexcept
{
    return value;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF KeyT<T, BB, IB>::operator T() noexcept
{
    return value;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
bool KeyT<T, BB, IB>::operator==(const KeyT& rhs) const noexcept
{
    return (value == rhs.value);
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
bool KeyT<T, BB, IB>::operator<(const KeyT& rhs) const noexcept
{
    return (value < rhs.value);
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
T KeyT<T, BB, IB>::FetchBatchPortion() const noexcept
{
    return (value & BatchMask) >> IdBits;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
T KeyT<T, BB, IB>::FetchIndexPortion() const noexcept
{
    return value & IdMask;
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
KeyT<T, BB, IB> KeyT<T, BB, IB>::CombinedKey(T batch, T id) noexcept
{
    assert(batch <= BatchMask);
    assert(id <= IdMask);
    return KeyT((batch << IdBits) | (id & IdMask));
}

template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
MR_PF_DEF
KeyT<T, BB, IB> KeyT<T, BB, IB>::InvalidKey() noexcept
{
    return KeyT(std::numeric_limits<T>::max());
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
TriKeyT<T, FB, BB, IB>::TriKeyT(T v) noexcept
    : value(v)
{}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
TriKeyT<T, FB, BB, IB>::operator T() const noexcept
{
    return value;
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
TriKeyT<T, FB, BB, IB>::operator T() noexcept
{
    return value;
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
bool TriKeyT<T, FB, BB, IB>::operator==(const TriKeyT& rhs) const noexcept
{
    return (value == rhs.value);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
bool TriKeyT<T, FB, BB, IB>::operator<(const TriKeyT& rhs) const noexcept
{
    return (value < rhs.value);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
T TriKeyT<T, FB, BB, IB>::FetchFlagPortion() const noexcept
{
    return (value & FlagMask) >> (BatchBits + IdBits);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
T TriKeyT<T, FB, BB, IB>::FetchBatchPortion() const noexcept
{
    return (value & BatchMask) >> IdBits;
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
T TriKeyT<T, FB, BB, IB>::FetchIndexPortion() const noexcept
{
    return (value & IdMask);
}

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
MR_PF_DEF
TriKeyT<T, FB, BB, IB> TriKeyT<T, FB, BB, IB>::CombinedKey(T flag, T batch, T id) noexcept
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
MR_PF_DEF
TriKeyT<T, FB, BB, IB> TriKeyT<T, FB, BB, IB>::InvalidKey() noexcept
{
    return TriKeyT(std::numeric_limits<T>::max());
}


