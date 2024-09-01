#pragma once

#include "Key.h"
#include "TracerTypes.h"

#include <fmt/format.h>
#include <fmt/core.h>

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
struct HexKeyT : public KeyT<T, BB, IB>
{
    explicit HexKeyT(const KeyT<T, BB, IB>& k) : KeyT<T, BB, IB>(k)
    {}
};

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
struct HexTriKeyT : public TriKeyT<T, FB, BB, IB>
{
    explicit HexTriKeyT(const TriKeyT<T, FB, BB, IB>& k) : TriKeyT<T, FB, BB, IB>(k)
    {}
};

template<std::unsigned_integral T, uint32_t BB, uint32_t IB>
auto format_as(const HexKeyT<T, BB, IB>& k)
{
    std::string s = fmt::format("[{:0>{}X}|{:0>{}X}]",
                                k.FetchBatchPortion(), (BB + 3) / 4,
                                k.FetchIndexPortion(), (IB + 3) / 4);
    return s;
}

template<std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
auto format_as(const HexTriKeyT<T, FB, BB, IB>& k)
{
    std::string s = fmt::format("[{:0>{}b}|{:0>{}X}|{:0>{}X}]",
                                k.FetchFlagPortion(), FB,
                                k.FetchBatchPortion(), (BB + 3) / 4,
                                k.FetchIndexPortion(), (IB + 3) / 4);
    return s;
}

inline auto format_as(const HitKeyPack& hk)
{
    return Tuple(HexKeyT(hk.accelKey), HexTriKeyT(hk.lightOrMatKey),
                 HexKeyT(hk.primKey), HexKeyT(hk.transKey));
}

inline auto format_as(const RayGMem& r)
{
    return Tuple(r.dir, r.pos, Vector2(r.tMin, r.tMax));
}