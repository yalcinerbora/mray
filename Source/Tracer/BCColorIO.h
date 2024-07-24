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
namespace BlockCompressedIO::BC1
{

    MRAY_GPU MRAY_GPU_INLINE
    Pair<Vector3, Vector3> ExtractColorsBC1(Vector2ui block)
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

    Vector2ui InjectColorsBC1(Vector2ui block, Pair<Vector3, Vector3> colorIn)
    {
        auto ComposeColor565 = [](Vector3 color) -> uint32_t
        {
            using Bit::NormConversion::ToUNormVarying;
            uint32_t r = ToUNormVarying<uint32_t, 5>(color[0]);
            uint32_t g = ToUNormVarying<uint32_t, 6>(color[1]);
            uint32_t b = ToUNormVarying<uint32_t, 5>(color[2]);
            return Bit::Compose<5u, 6u, 5u>(r, g, b);
        };

        uint32_t color0 = ComposeColor565(colorIn.first);
        uint32_t color1 = ComposeColor565(colorIn.second);
        uint32_t colorOut = Bit::Compose<16u, 16u>(color0, color1);
        return Vector2ui(colorOut, block[1]);
    }
}

namespace BlockCompressedIO::BC6H
{
    class ModeLookup
    {
        private:
        static constexpr std::array<uint32_t, 14> MODE_TABLE =
        {
            0b00000, 0b00001, 0b00010, 0b00110,
            0b01010, 0b01110, 0b10010, 0b10110,
            0b11010, 0b11110, 0b00011, 0b00111,
            0b01011, 0b01111
        };
        uint32_t mode;

        public:
        ModeLookup(uint32_t m) : mode(m) {}
        template<unsigned int... I>
        bool IsMode()
        {
            return ((mode == MODE_TABLE[I - 1]) || ...);
        }
    };

    // Found a pattern for w part
    // but the other part (xyz parts)
    // are quite scrambled
    template<uint32_t O>
    MRAY_GPU MRAY_GPU_INLINE
    uint32_t BisectWhite(uint64_t low, ModeLookup m)
    {
        auto FSB = [](uint32_t v, uint32_t start, uint32_t end)
        {
            return FetchSubPortion<uint32_t>(v, {start, end});
        };
        static_assert(O == 0 || O == 10 || O == 20, "Wrong offset");
        constexpr bool IsRed    = (O == 0);
        constexpr bool IsGreen  = (O == 10);
        constexpr bool IsBlue   = (O == 20);

        uint32_t w = FSB(block[0], 5 + O, 15 + O);
        w = (m.IsMode<2>())         ? (w & 0x003F) : w;
        w = (m.IsMode<6>())         ? (w & 0x00FF) : w;
        w = (m.IsMode<10>())        ? (w & 0x001F) : w;
        w = (m.IsMode<7, 8, 9>())   ? (w & 0x007F) : w;
        // Extra bits Middle
        if constexpr(IsRed)
        {
            w = (m.IsMode<4, 5>())  ? (w |= FSB(block[0], 39 + O, 40 + O) << 10u) : w;
            w = (m.IsMode<3>())     ? (w |= FSB(block[0], 40 + O, 41 + O) << 10u) : w;
        }
        else if constexpr(IsGreen)
        {
            w = (m.IsMode<3, 5>())  ? (w |= FSB(block[0], 39 + O, 40 + O) << 10u) : w;
            w = (m.IsMode<4>())     ? (w |= FSB(block[0], 40 + O, 41 + O) << 10u) : w;
        }
        else
        {
            w = (m.IsMode<3, 4>())  ? (w |= FSB(block[0], 39 + O, 50 + O) << 10u) : w;
            w = (m.IsMode<5>())     ? (w |= FSB(block[0], 40 + O, 41 + O) << 10u) : w;
        }
        // Extra bits End
        w = (m.IsMode<12, 13, 14>())    ? (w |= FSB(block[0], 44 + O, 45 + O) << 10u) : w;
        // Mode 13 and 14 are hipsters, bits are reversed
        w = (m.IsMode<13, 14>())        ? (w |= FSB(block[0], 43 + O, 44 + O) << 11u) : w;
        // 4 bit left, just read bit by bit
        // TODO: check "Bit twiddling hacks" for a cooler way maybe?
        w = (m.IsMode<14>())    ? (w |= FSB(block[0], 42 + O, 43 + O) << 12u) : w;
        w = (m.IsMode<14>())    ? (w |= FSB(block[0], 41 + O, 42 + O) << 13u) : w;
        w = (m.IsMode<14>())    ? (w |= FSB(block[0], 40 + O, 41 + O) << 14u) : w;
        w = (m.IsMode<14>())    ? (w |= FSB(block[0], 39 + O, 40 + O) << 15u) : w;
        return w;
    }

    MRAY_GPU MRAY_GPU_INLINE
    Pair<Vector3, Vector3> ExtractColorsBC1(Vector2ul block)
    {
        using namespace Bit;
        // Alias a function because it will be used a lot..
        auto FSB = [](uint32_t v, uint32_t start, uint32_t end)
        {
            return FetchSubPortion<uint32_t>(v, {start, end});
        };

        // One of the complex ones
        // Lets start with mode
        // mode can be either 2 or 5 bits
        uint32_t mode = FSB(block[0], 0, 2);
        mode = (mode < 2) ? mode : FSB(block[0], 0, 5);
        ModeLookup m(mode);

        // When looking at the table, this probably
        // optimized via some chip wire routing optimizer
        // **then** made it into a standard (prob Microsoft
        // for a XBOX maybe?)
        //
        // There should be some pattern to abuse but
        // lets write it to check.
        uint32_t rw = BisectWhite< 0>(block[0], m);
        uint32_t gw = BisectWhite<10>(block[0], m);
        uint32_t bw = BisectWhite<20>(block[0], m);



    }
}