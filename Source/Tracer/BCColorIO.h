#pragma once

#include "Core/Vector.h"
#include "Core/BitFunctions.h"

// I was going to call it a day and do not support this functionality
// after a little bit of digging old (BC1-5) bc compressions have straightforward packing
// scheme so we will color convert BC formats on the fly.
// Let's go!
//
// From Here
// https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression
// https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format
// https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format
// https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference
// Bit diagrams makes sense but I could not understand BC6H properly
// so this implementation is somewhat based on this impl
// https://github.com/iOrange/bcdec/blob/main/bcdec.h
//
// With this functionality renderer with any color space can use any texture
// (given it is compressed BC1-7) without and preprocessing.

namespace BlockCompressedIO
{
    // TODO: Too much registers (local spills) maybe?
    using ColorPack = StaticVector<Vector3, 6>;
}

namespace BlockCompressedIO::BC1
{
    MRAY_GPU
    ColorPack   ExtractColorsBC1(Vector2ui block);
    MRAY_GPU
    Vector2ui   InjectColorsBC1(Vector2ui block, const ColorPack& colorIn);
}

namespace BlockCompressedIO::BC6H
{
    //class ModeLookup
    //{
    //    private:
    //    static constexpr std::array<uint32_t, 14> MODE_TABLE =
    //    {
    //        0b00000, 0b00001, 0b00010, 0b00110,
    //        0b01010, 0b01110, 0b10010, 0b10110,
    //        0b11010, 0b11110, 0b00011, 0b00111,
    //        0b01011, 0b01111
    //    };
    //    uint32_t mode;

    //    public:
    //    ModeLookup(uint32_t m) : mode(m) {}
    //    template<unsigned int... I>
    //    bool IsMode()
    //    {
    //        return ((mode == MODE_TABLE[I - 1]) || ...);
    //    }
    //};

    //// Alias a function because it will be used a lot..
    //uint32_t FSB(uint32_t v, uint32_t start, uint32_t end)
    //{
    //    using namespace Bit;
    //    return FetchSubPortion<uint32_t>(v, {start, end});
    //};


    //// Found a pattern for w part
    //// but the other part (xyz parts)
    //// are quite scrambled
    //template<uint32_t O>
    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectWhite(uint64_t low, ModeLookup m)
    //{
    //    static_assert(O == 0 || O == 10 || O == 20, "Wrong offset");
    //    constexpr bool IsRed    = (O == 0);
    //    constexpr bool IsGreen  = (O == 10);
    //    constexpr bool IsBlue   = (O == 20);

    //    uint32_t w = FSB(low, 5 + O, 15 + O);
    //    w = (m.IsMode<2>())         ? (w & 0x003F) : w;
    //    w = (m.IsMode<6>())         ? (w & 0x00FF) : w;
    //    w = (m.IsMode<10>())        ? (w & 0x001F) : w;
    //    w = (m.IsMode<7, 8, 9>())   ? (w & 0x007F) : w;
    //    // Extra bits Middle
    //    if constexpr(IsRed)
    //    {
    //        w = (m.IsMode<4, 5>())  ? (w |= FSB(block[0], 39 + O, 40 + O) << 10u) : w;
    //        w = (m.IsMode<3>())     ? (w |= FSB(block[0], 40 + O, 41 + O) << 10u) : w;
    //    }
    //    else if constexpr(IsGreen)
    //    {
    //        w = (m.IsMode<3, 5>())  ? (w |= FSB(block[0], 39 + O, 40 + O) << 10u) : w;
    //        w = (m.IsMode<4>())     ? (w |= FSB(block[0], 40 + O, 41 + O) << 10u) : w;
    //    }
    //    else
    //    {
    //        w = (m.IsMode<3, 4>())  ? (w |= FSB(block[0], 39 + O, 50 + O) << 10u) : w;
    //        w = (m.IsMode<5>())     ? (w |= FSB(block[0], 40 + O, 41 + O) << 10u) : w;
    //    }
    //    // Extra bits End
    //    w = (m.IsMode<12, 13, 14>())    ? (w |= FSB(block[0], 44 + O, 45 + O) << 10u) : w;
    //    // Mode 13 and 14 are hipsters, bits are reversed
    //    w = (m.IsMode<13, 14>())        ? (w |= FSB(block[0], 43 + O, 44 + O) << 11u) : w;
    //    // 4 bit left, just read bit by bit
    //    // TODO: check "Bit twiddling hacks" for a cooler way maybe?
    //    w = (m.IsMode<14>())    ? (w |= FSB(block[0], 42 + O, 43 + O) << 12u) : w;
    //    w = (m.IsMode<14>())    ? (w |= FSB(block[0], 41 + O, 42 + O) << 13u) : w;
    //    w = (m.IsMode<14>())    ? (w |= FSB(block[0], 40 + O, 41 + O) << 14u) : w;
    //    w = (m.IsMode<14>())    ? (w |= FSB(block[0], 39 + O, 40 + O) << 15u) : w;
    //    return w;
    //}

    //template<uint32_t O>
    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectX(uint64_t low, ModeLookup m)
    //{
    //    static_assert(O == 0 || O == 10 || O == 20, "Wrong offset");
    //    constexpr bool IsRed = (O == 0);
    //    constexpr bool IsGreen = (O == 10);
    //    constexpr bool IsBlue = (O == 20);

    //    uint32_t x = FSB(low, 35 + O, 44 + O);
    //    x = (m.IsMode<1, 6, 8, 9>())    ? (x & 0x001F) : x;
    //    x = (m.IsMode<2, 7, 10>())      ? (x & 0x003F) : x;
    //    x = (m.IsMode<14>())            ? (x & 0x000F) : x;
    //    x = (m.IsMode<13>())            ? (x & 0x00FF) : x;
    //    // Permuted modes
    //    if constexpr(IsRed)
    //    {
    //        x = (m.IsMode<3>())     ? (x & 0x001F) : x;
    //        x = (m.IsMode<4, 5>())  ? (x & 0x000F) : x;
    //    }
    //    else if constexpr(IsGreen)
    //    {
    //        x = (m.IsMode<4>())     ? (x & 0x001F) : x;
    //        x = (m.IsMode<3, 5>())  ? (x & 0x000F) : x;
    //    }
    //    else
    //    {
    //        x = (m.IsMode<5>())     ? (x & 0x001F) : x;
    //        x = (m.IsMode<3, 4>())  ? (x & 0x000F) : x;
    //    }
    //    return x;
    //}

    //template<uint32_t O>
    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectRYZ(Vector2ul block, ModeLookup m)
    //{
    //    static_assert(O == 0 || O == 6, "Wrong offset");
    //    uint32_t yz = FSB(block[1], 0 + O, 6 + O);
    //    yz = (m.IsMode<1, 3, 6, 8, 9>())    ? (yz & 0x001F) : yz;
    //    yz = (m.IsMode<4, 5>())             ? (yz & 0x000F) : yz;
    //    // No ry or rz
    //    yz = (m.IsMode<11, 12, 13, 14>())   ? 0 : yz;
    //    return yz;
    //};

    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectGY(Vector2ul block, ModeLookup m)
    //{
    //    if(m.IsMode<11, 12, 13, 14>()) return 0;

    //    uint32_t gy = FSB(block[0], 41, 45);
    //    // Extra bits
    //    gy = (m.IsMode<2,6,7,8,9,10>()) ? (gy |= FSB(block[0], 24, 25) << 4u) : gy;
    //    gy = (m.IsMode<1>())            ? (gy |= FSB(block[0],  2,  3) << 4u) : gy;
    //    gy = (m.IsMode<2>())            ? (gy |= FSB(block[0],  2,  3) << 5u) : gy;
    //    gy = (m.IsMode<4>())            ? (gy |= FSB(block[1], 14, 15) << 5u) : gy;
    //    return gy;
    //};

    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectGZ(Vector2ul block, ModeLookup m)
    //{
    //    if (m.IsMode<11, 12, 13, 14>()) return 0;

    //    uint32_t gz = FSB(block[0], 51, 55);
    //    // Extra bits
    //    gz = (m.IsMode<1, 4, 6, 8, 9>())    ? (gz |= FSB(block[0], 40, 41) << 4u) : gz;
    //    gz = (m.IsMode<2>())                ? (gz |= FSB(block[0],  3,  5) << 4u) : gz;
    //    gz = (m.IsMode<7>())                ? (gz |= FSB(block[0], 13, 14) << 4u) : gz;
    //    gz = (m.IsMode<10>())               ? (gz |= FSB(block[0], 10, 11) << 4u) : gz;
    //    return gz;
    //};

    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectBY(Vector2ul block, ModeLookup m)
    //{
    //    if(m.IsMode<11, 12, 13, 14>()) return 0;

    //    uint32_t by = FSB(block[0], 61, 64);
    //    // Extra bits
    //    by = (m.IsMode<2,6,7,8,9,10>()) ? (by |= FSB(block[0], 14, 15) << 4u) : by;
    //    by = (m.IsMode<1>())            ? (by |= FSB(block[0],  3,  4) << 4u) : by;
    //    by = (m.IsMode<2, 10>())        ? (by |= FSB(block[0], 22, 23) << 5u) : by;
    //    return by;
    //};

    //MRAY_GPU MRAY_GPU_INLINE
    //uint32_t BisectBZ(Vector2ul block, ModeLookup m)
    //{
    //    if (m.IsMode<11, 12, 13, 14>()) return 0;

    //    uint32_t bz = FSB(block[0], 51, 55);
    //    // Extra bits
    //    bz = (m.IsMode<1, 4, 6, 8, 9>())    ? (bz |= FSB(block[0], 40, 41) << 4u) : bz;
    //    bz = (m.IsMode<2>())                ? (bz |= FSB(block[0],  3,  5) << 4u) : bz;
    //    bz = (m.IsMode<7>())                ? (bz |= FSB(block[0], 13, 14) << 4u) : bz;
    //    bz = (m.IsMode<10>())               ? (bz |= FSB(block[0], 10, 11) << 4u) : bz;
    //    return bz;
    //};

    //MRAY_GPU MRAY_GPU_INLINE
    //Pair<Vector3, Vector3> ExtractColorsBC1(Vector4ui block4C)
    //{
    //    Vector2ul block;
    //    block[0] = Bit::Compose<32, 32>(uint64_t(block4C[0]), uint64_t(block4C[1]));
    //    block[1] = Bit::Compose<32, 32>(uint64_t(block4C[2]), uint64_t(block4C[3]));

    //    // One of the complex ones
    //    // Lets start with mode
    //    // mode can be either 2 or 5 bits
    //    uint32_t mode = FSB(block[0], 0, 2);
    //    mode = (mode < 2) ? mode : FSB(block[0], 0, 5);
    //    ModeLookup m(mode);

    //    // When looking at the table, this probably
    //    // optimized via some chip wire routing optimizer
    //    // **then** made it into a standard (prob Microsoft
    //    // for a XBOX maybe?)
    //    //
    //    // There should be some pattern to abuse but
    //    // lets write it to check.
    //    uint32_t rw = BisectWhite< 0>(block[0], m);
    //    uint32_t gw = BisectWhite<10>(block[0], m);
    //    uint32_t bw = BisectWhite<20>(block[0], m);
    //    //
    //    uint32_t rx = BisectX< 0>(block[0], m);
    //    uint32_t gx = BisectX<10>(block[0], m);
    //    uint32_t bx = BisectX<20>(block[0], m);
    //    //
    //    uint32_t ry = BisectRYZ<0>(block, m);
    //    uint32_t rz = BisectRYZ<6>(block, m);
    //    //
    //    uint32_t gy = BisectGY(block, m);
    //    uint32_t gz = BisectGZ(block, m);
    //    //
    //    uint32_t by = BisectBY(block, m);
    //    uint32_t bz = BisectBZ(block, m);



    //}
}


namespace BlockCompressedIO
{

MRAY_GPU MRAY_GPU_INLINE
ColorPack BC1::ExtractColorsBC1(Vector2ui block)
{
    auto BisectColor565 = [](uint16_t color) -> Vector3
    {
        using Bit::NormConversion::FromUNormVarying;
        uint16_t r = Bit::FetchSubPortion(color, {0, 5});
        uint16_t g = Bit::FetchSubPortion(color, {5, 11});
        uint16_t b = Bit::FetchSubPortion(color, {11, 16});
        return Vector3(FromUNormVarying<Float, 5>(r),
                        FromUNormVarying<Float, 6>(g),
                        FromUNormVarying<Float, 5>(b));

    };
    uint16_t color0 = uint16_t(Bit::FetchSubPortion(block[0], {0, 16}));
    uint16_t color1 = uint16_t(Bit::FetchSubPortion(block[0], {16, 32}));
    return {BisectColor565(color0),BisectColor565(color1)};
}

MRAY_GPU MRAY_GPU_INLINE
Vector2ui BC1::InjectColorsBC1(Vector2ui block, const ColorPack& colorIn)
{
    auto ComposeColor565 = [](Vector3 color) -> uint32_t
    {
        using Bit::NormConversion::ToUNormVarying;
        uint32_t r = ToUNormVarying<uint32_t, 5>(color[0]);
        uint32_t g = ToUNormVarying<uint32_t, 6>(color[1]);
        uint32_t b = ToUNormVarying<uint32_t, 5>(color[2]);
        return Bit::Compose<5u, 6u, 5u>(r, g, b);
    };

    uint32_t color0 = ComposeColor565(colorIn[0]);
    uint32_t color1 = ComposeColor565(colorIn[1]);
    uint32_t colorOut = Bit::Compose<16u, 16u>(color0, color1);
    return Vector2ui(colorOut, block[1]);
}

}