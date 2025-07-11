#pragma once

#include "Key.h"
#include "TracerTypes.h"

#include "Core/MathForward.h"

#include <fmt/format.h>
#include <fmt/core.h>

// Uncomment when needed
// #include "AcceleratorLinear.h"
//template <> struct fmt::formatter<LBVHAccelDetail::LBVHNode> : formatter<std::string>
//{
//    auto format(LBVHAccelDetail::LBVHNode, format_context& ctx) const
//        ->format_context::iterator;
//};
//
//inline auto fmt::formatter<LBVHAccelDetail::LBVHNode>::format(LBVHAccelDetail::LBVHNode n,
//                                                              format_context& ctx) const
//    -> format_context::iterator
//{
//    std::string out = MRAY_FORMAT("[L{}, R{}, P:{}]",
//                                  HexKeyT(n.leftIndex), HexKeyT(n.rightIndex), n.parentIndex);
//    return formatter<std::string>::format(out, ctx);
//}

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

// Rely on overload resolution to write in hexadecimal
template <std::unsigned_integral T, uint32_t BB, uint32_t IB>
struct HexKeyT : public KeyT<T, BB, IB>
{
    HexKeyT() = default;
    explicit HexKeyT(const KeyT<T, BB, IB>& k) : KeyT<T, BB, IB>(k)
    {}
};

template <std::unsigned_integral T, uint32_t FB, uint32_t BB, uint32_t IB>
struct HexTriKeyT : public TriKeyT<T, FB, BB, IB>
{
    HexTriKeyT() = default;
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
    return std::tuple(HexKeyT(hk.primKey), HexTriKeyT(hk.lightOrMatKey),
                      HexKeyT(hk.transKey), HexKeyT(hk.accelKey));
}

inline auto format_as(const RayGMem& r)
{
    return std::tuple(r.dir, r.pos, Vector2(r.tMin, r.tMax));
}

inline auto format_as(const ImageCoordinate& ic)
{
    Vector2 total = ic.GetPixelIndex();
    Vector2 offset = Vector2(ic.offset);
    std::string s = fmt::format("[{} | {} = {}]",
                                ic.pixelIndex, offset, total);
    return s;
}