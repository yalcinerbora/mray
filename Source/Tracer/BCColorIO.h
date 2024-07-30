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
    struct BC1
    {
        using ColorPack = std::array<Vector3, 2>;
        using BlockType = Vector2ui;

        MRAY_HYBRID
        static ColorPack    ExtractColors(Vector2ui block);
        MRAY_HYBRID
        static Vector2ui    InjectColors(Vector2ui block, const ColorPack& colorIn,
                                         bool skipAlpha = false);
    };

    struct BC2
    {
        using ColorPack = std::array<Vector3, 2>;
        using BlockType = Vector4ui;

        MRAY_HYBRID
        static ColorPack    ExtractColors(Vector4ui block);
        MRAY_HYBRID
        static Vector4ui    InjectColors(Vector4ui block, const ColorPack& colorIn);
    };

    struct BC3
    {
        using ColorPack = std::array<Vector3, 2>;
        using BlockType = Vector4ui;

        MRAY_HYBRID
        static ColorPack   ExtractColors(Vector4ui block);
        MRAY_HYBRID
        static Vector4ui   InjectColors(Vector4ui block, const ColorPack& colorIn);
    };

    template<bool IsSigned>
    struct BC4
    {
        using ColorPack = std::array<Vector3, 2>;
        using BlockType = Vector2ui;

        MRAY_HYBRID
        static ColorPack    ExtractColors(Vector2ui block);
        MRAY_HYBRID
        static Vector2ui    InjectColors(Vector2ui block, const ColorPack& colorIn);
    };

    template<bool IsSigned>
    struct BC5
    {
        using ColorPack = std::array<Vector3, 2>;
        using BlockType = Vector4ui;

        MRAY_HYBRID
        static ColorPack    ExtractColors(Vector4ui block);
        MRAY_HYBRID
        static Vector4ui    InjectColors(Vector4ui block, const ColorPack& colorIn);
    };

    template<bool IsSigned>
    struct BC6H
    {
        using ColorPack = std::array<Vector3, 4>;
        using BlockType = Vector4ui;

        MRAY_HYBRID
        static ColorPack    ExtractColors(Vector4ui block);
        MRAY_HYBRID
        static Vector4ui    InjectColors(Vector4ui block, const ColorPack& colorIn);
    };

    struct BC7
    {
        public:
        using BlockType = Vector4ui;

        private:
        Vector2ul   block;

        MRAY_HYBRID
        uint32_t    Mode() const;
        MRAY_HYBRID
        uint32_t    ColorStartOffset(uint32_t mode) const;
        MRAY_HYBRID
        uint32_t    ColorBits(uint32_t mode) const;
        MRAY_HYBRID
        uint32_t    ColorCount(uint32_t mode) const;
        MRAY_HYBRID
        bool        HasUniquePBits(uint32_t mode) const;

        public:
        MRAY_HYBRID BC7(const Vector4ui block);

        MRAY_HYBRID
        Vector4ui   Block() const;

        MRAY_HYBRID
        Vector3     ExtractColor(uint32_t i) const;
        MRAY_HYBRID
        void        InjectColor(uint32_t i, const Vector3& colorIn);
        MRAY_HYBRID
        uint32_t    ColorCount() const;
    };
}

namespace BlockCompressedIO
{

MRAY_HYBRID MRAY_CGPU_INLINE
typename BC1::ColorPack
BC1::ExtractColors(Vector2ui block)
{
    auto BisectColor565 = [](uint32_t color) -> Vector3
    {
        using Bit::NormConversion::FromUNormVarying;
        uint32_t b = Bit::FetchSubPortion(color, {0, 5});
        uint32_t g = Bit::FetchSubPortion(color, {5, 11});
        uint32_t r = Bit::FetchSubPortion(color, {11, 16});
        return Vector3(FromUNormVarying<Float>(r, 5u),
                       FromUNormVarying<Float>(g, 6u),
                       FromUNormVarying<Float>(b, 5u));

    };
    uint32_t color0 = Bit::FetchSubPortion(block[0], {0, 16});
    uint32_t color1 = Bit::FetchSubPortion(block[0], {16, 32});
    return { BisectColor565(color0), BisectColor565(color1) };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2ui BC1::InjectColors(Vector2ui block, const ColorPack& colorIn,
                            bool skipAlpha)
{
    auto ComposeColor565 = [](Vector3 color) -> uint32_t
    {
        color.ClampSelf(Float(0), Float(1));
        using Bit::NormConversion::ToUNormVarying;
        uint32_t r = ToUNormVarying<uint32_t>(color[0], 5);
        uint32_t g = ToUNormVarying<uint32_t>(color[1], 6);
        uint32_t b = ToUNormVarying<uint32_t>(color[2], 5);
        return Bit::Compose<5u, 6u, 5u>(b, g, r);
    };

    uint32_t color0 = ComposeColor565(colorIn[0]);
    uint32_t color1 = ComposeColor565(colorIn[1]);
    // 1-bit alpha mode
    // https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/opaque-and-1-bit-alpha-textures
    uint32_t c0 = Bit::FetchSubPortion(block[0], {0, 16});
    uint32_t c1 = Bit::FetchSubPortion(block[0], {16, 32});
    bool opaqueMode = (c0 > c1);
    bool alphaMode = !opaqueMode;
    // Due to conversion, new color's magnitudes are changed
    // obey the actual mode
    if(!skipAlpha)
    {
        // Was alpha mode, new colors
        // seem opaque mode. Just swap the colors
        // c_2 is middle of c_0 c_1 and c_3 is black
        if(alphaMode && color0 > color1)
            std::swap(color0, color1);

        // Opposite case, was opaque now alpha
        else if(opaqueMode && color0 == color1)
            // For equal case we can set lookup
            // table to zero.
            block[1] = 0x0000;
        else if(opaqueMode && color0 < color1)
        {
            // Hard part we can not do swap etc. here
            // (We can swap but we need to swap the lookup
            // table as well.
            // So what do we do? Swap
            // This is not correct, but we assume colors are
            // closeby.
            // TODO: Change this later
            std::swap(color0, color1);

        }
    }
    uint32_t block0Out = Bit::Compose<16u, 16u>(color0, color1);
    return Vector2ui(block0Out, block[1]);
}

MRAY_HYBRID MRAY_CGPU_INLINE
typename BC2::ColorPack
BC2::ExtractColors(Vector4ui block)
{
    return BC1::ExtractColors(Vector2ui(block[2], block[3]));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector4ui BC2::InjectColors(Vector4ui block, const ColorPack& colorIn)
{
    auto b0 = BC1::InjectColors(Vector2ui(block[2], block[3]), colorIn,
                                true);
    return Vector4ui(block[0], block[1], b0[0], block[3]);
}

MRAY_HYBRID MRAY_CGPU_INLINE
typename BC3::ColorPack
BC3::ExtractColors(Vector4ui block)
{
    return BC1::ExtractColors(Vector2ui(block[2], block[3]));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector4ui BC3::InjectColors(Vector4ui block, const ColorPack& colorIn)
{
    auto b0 = BC1::InjectColors(Vector2ui(block[2], block[3]), colorIn,
                                true);
    return Vector4ui(block[0], block[1], b0[0], block[3]);
}

template <bool IsSigned>
MRAY_HYBRID MRAY_CGPU_INLINE
typename BC4<IsSigned>::ColorPack
BC4<IsSigned>::ExtractColors(Vector2ui block)
{
    using namespace Bit;
    using namespace NormConversion;
    uint8_t color0 = uint8_t(FetchSubPortion(block[0], {0, 8}));
    uint8_t color1 = uint8_t(FetchSubPortion(block[0], {8, 16}));

    Float c0, c1;
    if constexpr(IsSigned)
    {
        c0 = FromSNorm<Float>(std::bit_cast<int8_t>(color0));
        c1 = FromSNorm<Float>(std::bit_cast<int8_t>(color1));
    }
    else
    {
        c0 = FromUNorm<Float>(color0);
        c1 = FromUNorm<Float>(color1);
    }
    return {Vector3(c0, 0, 0), Vector3(c1, 0, 0)};
}

template <bool IsSigned>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector2ui BC4<IsSigned>::InjectColors(Vector2ui block, const ColorPack& colorIn)
{
    using namespace Bit;
    using namespace NormConversion;
    using MathFunctions::Clamp;

    uint32_t c0 = FetchSubPortion(block[0], {0, 8});
    uint32_t c1 = FetchSubPortion(block[0], {8, 16});
    // Due to conversion new color's magnitudes
    // may have changed. Obey the actual mode.
    bool opaqueMode = (c0 > c1);
    bool alphaMode = !opaqueMode;

    uint32_t color0, color1;
    if constexpr(IsSigned)
    {
        Float r0 = Clamp<Float>(colorIn[0][0], -1, 1);
        Float r1 = Clamp<Float>(colorIn[1][0], -1, 1);
        color0 = std::bit_cast<uint8_t>(ToSNorm<int8_t>(r0));
        color1 = std::bit_cast<uint8_t>(ToSNorm<int8_t>(r1));
    }
    else
    {
        Float r0 = Clamp<Float>(colorIn[0][0], 0, 1);
        Float r1 = Clamp<Float>(colorIn[1][0], 0, 1);
        color0 = ToUNorm<uint8_t>(r0);
        color1 = ToUNorm<uint8_t>(r1);
    }

    // If mode is changed, try to convert back to mode
    // It was alpha (4 color) mode now it is (6 color)
    if(alphaMode && color0 > color1)
    {
        // Unlike BC1 we can not swap here?
        // TODO: Change this
        std::swap(color0, color1);
    }
    else if(opaqueMode && color0 == color1)
    {
        // This is somewhat easy.
        // Either decrement or increment
        if(color0 != 0) color0--;
        else color1++;
    }
    if(opaqueMode && color0 < color1)
    {
        // Unlike BC1 we can not swap here?
        // TODO: Change this
        std::swap(color0, color1);
    }
    uint32_t block0 = Compose<8, 8, 16>(c0, c1, FetchSubPortion(block[0], {16, 32}));
    return Vector2ui(block0, block[1]);
}

template<bool IsSigned>
MRAY_HYBRID MRAY_CGPU_INLINE
typename BC5<IsSigned>::ColorPack
BC5<IsSigned>::ExtractColors(Vector4ui block)
{
    auto [r0, r1] = BC4<IsSigned>::ExtractColors(Vector2ui(block[0], block[1]));
    auto [g0, g1] = BC4<IsSigned>::ExtractColors(Vector2ui(block[2], block[3]));

    return {Vector3(r0[0], g0[0], 0), Vector3(r1[0], g1[0], 0)};
}

template<bool IsSigned>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector4ui BC5<IsSigned>::InjectColors(Vector4ui block, const ColorPack& colorIn)
{
    using BC4ColorPack = typename BC4<IsSigned>::ColorPack;
    BC4ColorPack r = {Vector3(colorIn[0][0], 0 ,0), Vector3(colorIn[1][0], 0 ,0)};
    BC4ColorPack g = {Vector3(colorIn[0][1], 0 ,0), Vector3(colorIn[1][1], 0 ,0)};
    auto b0 = BC4<IsSigned>::InjectColors(Vector2ui(block[0], block[1]), r);
    auto b1 = BC4<IsSigned>::InjectColors(Vector2ui(block[2], block[3]), g);
    return Vector4ui(b0[0], b0[1], b1[0], b1[1]);
}

template<bool IsSigned>
MRAY_HYBRID MRAY_CGPU_INLINE
typename BC6H<IsSigned>::ColorPack
BC6H<IsSigned>::ExtractColors(Vector4ui block)
{
    return {};
}

template<bool IsSigned>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector4ui BC6H<IsSigned>::InjectColors(Vector4ui block, const ColorPack& colorIn)
{
    return Vector4ui::Zero();
}

MRAY_HYBRID MRAY_CGPU_INLINE
BC7::BC7(const Vector4ui b)
    : block(Bit::Compose<32, 32>(uint64_t(b[0]), b[1]),
            Bit::Compose<32, 32>(uint64_t(b[2]), b[3]))
{}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t BC7::Mode() const
{
    uint32_t mode = Bit::CountTZero(uint32_t(block[0]));
    return std::min(mode, 8u);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t BC7::ColorStartOffset(uint32_t mode) const
{
    if(mode == 0)       return 1 + 4;
    else if(mode == 1)  return 2 + 6;
    else if(mode == 2)  return 3 + 6;
    else if(mode == 3)  return 4 + 6;
    else if(mode == 4)  return 5 + 3;
    else if(mode == 5)  return 6 + 2;
    else if(mode == 6)  return 7 + 0;
    else                return 8 + 6;

}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t BC7::ColorBits(uint32_t mode) const
{
    if(mode == 0)
        return 4;
    else if(mode == 1)
        return 6;
    else if(mode == 2 || mode == 4 || mode == 7)
        return 5;
    else
        return 7;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t BC7::ColorCount(uint32_t mode) const
{
    if(mode == 0 || mode == 2)
        return 6;
    else if(mode == 1 || mode == 3 || mode == 7)
        return 4;
    else
        return 2;
}

MRAY_HYBRID MRAY_CGPU_INLINE
bool BC7::HasUniquePBits(uint32_t mode) const
{
    return (mode == 0 || mode == 3 || mode == 6 || mode == 7);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector4ui BC7::Block() const
{
    return Vector4ui(block[0] & 0xFFFFFFFF, block[0] >> 32,
                     block[1] & 0xFFFFFFFF, block[1] >> 32);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 BC7::ExtractColor(uint32_t i) const
{
    auto InjectPBits = [](Vector3ui color, uint32_t bit) -> Vector3ui
    {
        assert(bit <= 1);
        return Vector3ui((color[0] << 1u) | bit,
                         (color[1] << 1u) | bit,
                         (color[2] << 1u) | bit);
    };

    auto ExpandAndRepeat = [](Vector3ui color, Vector3ui colorBits) -> Vector3ui
    {
        Vector3ui result(color[0] << (8u - colorBits[0]),
                         color[1] << (8u - colorBits[1]),
                         color[2] << (8u - colorBits[2]));
        result[0] |= (result[0] >> colorBits[0]);
        result[1] |= (result[1] >> colorBits[1]);
        result[2] |= (result[2] >> colorBits[2]);
        return result;
    };

    uint32_t mode = Mode();
    if(i >= ColorCount(mode)) return Vector3::Zero();

    // Rotation bits (only valid for mode 4 and 5)
    uint32_t rotation = uint32_t(Bit::FetchSubPortion(block[0], {mode + 1, mode + 3}));
    bool hasRotation = (mode == 5 || mode == 4) && (rotation != 0);

    // Cycle through bits and read
    const uint32_t CC = ColorCount(mode);
    const uint32_t B = ColorBits(mode);
    const uint32_t O = B * i;
    using Bit::FetchSubPortion;
    using Bit::RotateRight;
    auto b = RotateRight(block.AsArray(), ColorStartOffset(mode));

    // R,G,B
    Vector3ui c = Vector3ui::Zero();
    UNROLL_LOOP
    for(uint32_t idx = 0; idx < 3; idx++)
    {
        //if(!(hasRotation && rotation == (idx + 1)))
            c[idx] = uint32_t(FetchSubPortion(b[0], {O, O + B}));
        b = RotateRight(b, CC * B);
    }
    // Skip alpha row for mode 6 & 7
    if(mode == 6 || mode == 7)
        b = RotateRight(b, CC * B);
    // If has unique p-bits
    if(HasUniquePBits(mode))
    {
        uint32_t p = uint32_t(FetchSubPortion(b[0], {i, i + 1}));
        c = InjectPBits(c, p);
    }
    // Shared P-bits
    if(mode == 1)
    {
        // Every to color shares a single bit
        uint32_t o = i >> 1;
        uint32_t p = uint32_t(FetchSubPortion(b[0], {o, o + 1}));
        c = InjectPBits(c, p);
    }
    // Has rotation we need to update alpha with color
    Vector3ui cBitCount = Vector3ui::Zero();
    if(hasRotation)
    {
        // Alpha bits are one-bit larger so offset is different
        uint32_t o = (B + 1) * i;
        c[rotation - 1] = uint32_t(FetchSubPortion(b[0], {o, o + B + 1}));
        // This is little bit different than the generic
        // case migh as well return here
        uint32_t rb = (rotation == 1) ? (B + 1) : B;
        uint32_t gb = (rotation == 2) ? (B + 1) : B;
        uint32_t bb = (rotation == 3) ? (B + 1) : B;
        cBitCount = Vector3ui(rb, gb, bb);
    }
    else
    {
        bool expandChannel = (HasUniquePBits(mode) || mode == 1);
        uint32_t cBits = expandChannel ? (B + 1) : B;
        cBitCount = Vector3ui(cBits);
    }

    // Expand the bits and repeat MSB over LSB
    c = ExpandAndRepeat(c, cBitCount);
    using namespace Bit::NormConversion;
    return Vector3(FromUNorm<Float>(uint8_t(c[0])),
                   FromUNorm<Float>(uint8_t(c[1])),
                   FromUNorm<Float>(uint8_t(c[2])));
}

MRAY_HYBRID MRAY_CGPU_INLINE
void BC7::InjectColor(uint32_t i, const Vector3& colorIn)
{
    auto GetPBitAndPack = [](uint32_t& p, Vector3ui c) -> Vector3ui
    {
        // P-bit is on c[3]
        // TODO: Rounding up here?
        p = (c[0] & 0x1) + (c[1] & 0x1) + (c[2] & 0x1);
        p = (p + 2) / 4;
        return Vector3ui(c[0] >> 1u, c[1] >> 1u, c[2] >> 1u);
    };

    auto RoundAndPack = [](Vector3ui c, Vector3ui cBits) -> Vector3ui
    {
        // Round the value
        c[0] = std::min(c[0] + (1u << (8u - cBits[0])) - 1, 255u);
        c[1] = std::min(c[1] + (1u << (8u - cBits[1])) - 1, 255u);
        c[2] = std::min(c[2] + (1u << (8u - cBits[2])) - 1, 255u);
        // Get the MSB portion
        return Vector3ui(c[0] >> (8u - cBits[0]),
                         c[1] >> (8u - cBits[1]),
                         c[2] >> (8u - cBits[2]));
    };

    uint32_t mode = Mode();
    if(i >= ColorCount(mode)) return;

    const uint32_t CC = ColorCount(mode);
    const uint32_t B = ColorBits(mode);
    const uint32_t O = B * i;
    // Rotation bits (only valid for mode 4 and 5)
    uint32_t rotation = uint32_t(Bit::FetchSubPortion(block[0], {mode + 1, mode + 3}));
    bool hasRotation = (mode == 5 || mode == 4) && (rotation != 0);
    bool hasPBits = (HasUniquePBits(mode) || (mode == 1));

    Vector3ui cBitCount = Vector3ui::Zero();
    if(mode == 5 || mode == 4)
    {
        using namespace Bit::NormConversion;
        uint32_t colorBits = B;
        uint32_t rb = (rotation == 1) ? colorBits + 1 : colorBits;
        uint32_t gb = (rotation == 2) ? colorBits + 1 : colorBits;
        uint32_t bb = (rotation == 3) ? colorBits + 1 : colorBits;
        cBitCount = Vector3ui(rb, gb, bb);
    }
    else
    {
        uint32_t colorBits = (hasPBits) ? B + 1 : B;
        cBitCount = Vector3ui(colorBits);
    }
    // Convert back to full 8-bit
    using namespace Bit::NormConversion;
    Vector3ui c = Vector3ui(ToUNorm<uint8_t>(colorIn[0]),
                            ToUNorm<uint8_t>(colorIn[1]),
                            ToUNorm<uint8_t>(colorIn[2]));
    // Round the lower bits and pack to the specified bit width
    c = RoundAndPack(c, cBitCount);

    // Fetch pBit from the data
    uint32_t pBit = 0;
    if(hasPBits) c = GetPBitAndPack(pBit, c);

    // Cycle through bits and write
    using Bit::SetSubPortion;
    using Bit::RotateRight;
    auto b  = RotateRight(block.AsArray(), ColorStartOffset(mode));

    // R,G,B
    UNROLL_LOOP
    for(uint32_t idx = 0; idx < 3; idx++)
    {
        if(!(hasRotation && rotation == (idx + 1)))
            b[0] = SetSubPortion(b[0], c[idx], {O, O + B});

        b = RotateRight(b, CC * B);
    }
    // Skip alpha row for mode 6 & 7
    if(mode == 6 || mode == 7)
        b = RotateRight(b, CC * B);

    // If has unique p-bits
    if(HasUniquePBits(mode))
        b[0] = SetSubPortion(b[0], pBit, {i, i + 1});

    // Shared P-bits
    if(mode == 1)
    {
        uint32_t o = i >> 1;
        // Special case: Due to streaming,
        // We need to find the rounded bit between two different colors
        // However we do not store the previous bit (TODO: Later)
        // Fetch the current set bit and do some heuristic
        uint32_t pPrev = uint32_t(Bit::FetchSubPortion(b[0], {o, o + 1}));
        pBit = (pPrev + pBit) >> 1;
        b[0] = SetSubPortion(b[0], pBit, {o, o + 1});
    }
    // Has rotation we need to update alpha with color
    if(hasRotation)
    {
        uint32_t o = (B + 1) * i;
        b[0] = SetSubPortion(b[0], c[rotation - 1], {o, o + B + 1});
    }
    // Rotate the bits back to the proper location
    uint32_t ccbCount = (mode == 6 || mode == 7) ? 4 : 3;
    uint32_t rBack = (CC * B * ccbCount) + ColorStartOffset(mode);
    b = Bit::RotateRight(b, 128u - rBack);
    block = Vector2ul(b[0], b[1]);
}



MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t BC7::ColorCount() const
{
    return ColorCount(Mode());
}

}

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
