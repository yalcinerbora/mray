#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Types.h"
#include "Error.h"

namespace Color
{
    // Representation conversions
    MR_PF_DECL Vector3 HSVToRGB(const Vector3& hsv) noexcept;

    // Stateless random color,
    // Neighboring pixels should have distinct colors
    MR_PF_DECL Vector3 RandomColorRGB(uint32_t index) noexcept;

    // Color conversion related
    //
    struct Primaries
    {
        Vector2 xyRed;
        Vector2 xyGreen;
        Vector2 xyBlue;
        Vector2 xyWhite;
    };
    //
    MR_PF_DECL Vector3   XYZToYxy(const Vector3& xyz) noexcept;

    MR_PF_DECL Vector3   YxyToXYZ(const Vector3& yXY) noexcept;

    MR_PF_DECL Matrix3x3 BradfordMatrix() noexcept;

    MR_PF_DECL Matrix3x3 InvBradfordMatrix() noexcept;

    MR_PF_DECL Matrix3x3 GenWhitepointMatrix(const Vector2& fromWhite,
                                             const Vector2& toWhite) noexcept;
    MR_PF_DECL Vector3   GenWhitepointXYZ(const Vector2& xy) noexcept;

    MR_PF_DECL Matrix3x3 GenRGBToXYZ(const Primaries&) noexcept;
    MR_PF_DECL Matrix3x3 GenXYZToRGB(const Primaries&) noexcept;

    // https://en.wikipedia.org/wiki/Standard_illuminant
    // Some whitepoints are commented out, current support
    // is only for a couple of color spaces
    constexpr Vector2 D50Whitepoint = Vector2(0.34567, 0.35850);
    constexpr Vector2 D55Whitepoint = Vector2(0.33242, 0.34743);
    // These are different
    // https://docs.acescentral.com/tb/white-point#comparison-of-the-aces-white-point-and-cie-d60
    constexpr Vector2 D60Whitepoint = Vector2(0.32169, 0.33780);
    constexpr Vector2 ACESWhitepoint = Vector2(0.32168, 0.33767);
    constexpr Vector2 D65Whitepoint = Vector2(0.31272, 0.32903);
    constexpr Vector2 D75Whitepoint = Vector2(0.29902, 0.31485);

    MR_PF_DECL Primaries FindPrimaries(MRayColorSpaceEnum);

    // Colorspace guarantees whitepoints match
    // by enforcing every colorspace to convert to D65
    // (Only for ACES spaces)
    template <MRayColorSpaceEnum E>
    class Colorspace
    {
        private:
        static constexpr Primaries Prims = FindPrimaries(E);

        // Expose these for testing
        public:
        static constexpr Matrix3x3 ToXYZMatrix = (E == MRayColorSpaceEnum::MR_DEFAULT)
            ? Matrix3x3::Identity()
            : (GenWhitepointMatrix(Prims.xyWhite, D65Whitepoint) * GenRGBToXYZ(Prims));

        static constexpr Matrix3x3 FromXYZMatrix = ToXYZMatrix.Inverse();

        public:
        Colorspace() = default;

        MR_PF_DECL Vector3 ToXYZ(const Vector3&) const noexcept;
        MR_PF_DECL Vector3 FromXYZ(const Vector3&) const noexcept;
    };

    template <MRayColorSpaceEnum FromE, MRayColorSpaceEnum ToE>
    class ColorspaceTransfer
    {
        // Expose these for testing
        public:
        static constexpr Matrix3x3 RGBToRGBMatrix = (FromE == ToE)
            ? Matrix3x3::Identity()
            : (Colorspace<ToE>::FromXYZMatrix * Colorspace<FromE>::ToXYZMatrix);

        public:
        ColorspaceTransfer() = default;

        MR_PF_DECL Vector3 Convert(const Vector3& rgb) const noexcept;
    };

    // EOTF/OETF (Gamma) Related Conversions
    // Only basic gamma currently
    class OpticalTransferGamma
    {
        private:
        Float               gamma;

        public:
        // Constructors & Destructor
        MR_PF_DECL_V       OpticalTransferGamma(Float gamma) noexcept;
        // Only providing "To" function here since renderer works in
        // linear space. "From" function is on Visor
        MR_PF_DECL Vector3 ToLinear(const Vector3& color) const noexcept;
    };

    class OpticalTransferIdentity
    {
        public:
        MR_PF_DECL Vector3 ToLinear(const Vector3& color) const noexcept;
    };
}

MR_PF_DEF Vector3 Color::HSVToRGB(const Vector3& hsv) noexcept
{
    // H, S, V both normalized
    // H: [0-1) (meaning 0 is 0, 1 is 360)
    // S: [0-1] (meaning 0 is 0, 1 is 100)
    // V: [0-1] (meaning 0 is 0, 1 is 100)
    Float h = hsv[0] * Float(360);
    constexpr Float o60 = Float(1) / Float(60);

    Float c = hsv[2] * hsv[1];
    Float m = hsv[2] - c;
    Float x = Math::FMod(h * o60, Float(2));
    x = Math::Abs(x - Float(1));
    x = c * (Float(1) - x);

    Vector3 result;
    using I = IntegralSister<Float>;
    I sextant = static_cast<I>(h) / I(60) % I(6);
    switch(sextant)
    {
        case 0: result = Vector3(c, x, 0); break;
        case 1: result = Vector3(x, c, 0); break;
        case 2: result = Vector3(0, c, x); break;
        case 3: result = Vector3(0, x, c); break;
        case 4: result = Vector3(x, 0, c); break;
        case 5: result = Vector3(c, 0, x); break;
    }
    result = result + m;
    return result;
}

MR_PF_DEF Vector3 Color::RandomColorRGB(uint32_t index) noexcept
{
    // https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    // Specifically using double here to get some precision
    constexpr double SATURATION = 0.75;
    constexpr double VALUE = 0.95;
    constexpr double GOLDEN_RATIO_CONJ = 0.618033988749895;
    // For large numbers use double arithmetic here
    double hue = 0.1 + static_cast<double>(index) * GOLDEN_RATIO_CONJ;
    hue -= Math::Floor(hue);

    return HSVToRGB(Vector3(hue, SATURATION, VALUE));
}

MR_PF_DEF Vector3 Color::XYZToYxy(const Vector3& xyz) noexcept
{
    Float sum = (xyz[0] + xyz[1] + xyz[2]);
    Float invSum = (sum == Float(0)) ? Float(0) : Float(1) / sum;
    // This has slightly better precision maybe?
    Float x = Float(1) - (xyz[1] + xyz[2]) * invSum;
    Float y = Float(1) - (xyz[0] + xyz[2]) * invSum;
    return Vector3(xyz[1], x, y);
}

MR_PF_DEF Vector3 Color::YxyToXYZ(const Vector3& yXY) noexcept
{
    // https://www.easyrgb.com/en/math.php
    Float yy = (yXY[2] == Float(0)) ? Float(0) : (yXY[0] / yXY[2]);
    Float x = yXY[1] * yy;
    Float y = yXY[0];
    Float z = (Float(1) - yXY[1] - yXY[2]) * yy;
    return Vector3(x, y, z);
}

MR_PF_DEF Matrix3x3 Color::BradfordMatrix() noexcept
{
    return Matrix3x3( 0.8951f,  0.2664f, -0.1614f,
                     -0.7502f,  1.7135f,  0.0367f,
                      0.0389f, -0.0685f,  1.0296f);
}

MR_PF_DEF Matrix3x3 Color::InvBradfordMatrix() noexcept
{
    return BradfordMatrix().Inverse();
}

MR_PF_DEF Matrix3x3 Color::GenWhitepointMatrix(const Vector2& fromWhite,
                                               const Vector2& toWhite) noexcept
{
    Vector3 fwXYZ = YxyToXYZ(Vector3(1, fromWhite[0], fromWhite[1]));
    Vector3 twXYZ = YxyToXYZ(Vector3(1, toWhite[0], toWhite[1]));
    Vector3 r = twXYZ / fwXYZ;
    Matrix3x3 scaleMatrix = Matrix3x3(r[0], 0,    0,
                                      0,    r[1], 0,
                                      0,    0,    r[2]);
    return InvBradfordMatrix() * scaleMatrix * BradfordMatrix();
}

MR_PF_DEF Vector3 Color::GenWhitepointXYZ(const Vector2& xy) noexcept
{
    return YxyToXYZ(Vector3(1, xy[0], xy[1]));
}

MR_PF_DEF Matrix3x3 Color::GenRGBToXYZ(const Primaries& p) noexcept
{
    // http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    Vector3 xyzRed = YxyToXYZ(Vector3(1, p.xyRed[0], p.xyRed[1]));
    Vector3 xyzGreen = YxyToXYZ(Vector3(1, p.xyGreen[0], p.xyGreen[1]));
    Vector3 xyzBlue = YxyToXYZ(Vector3(1, p.xyBlue[0], p.xyBlue[1]));
    Vector3 xyzWhite = YxyToXYZ(Vector3(1, p.xyWhite[0], p.xyWhite[1]));

    // We are transposing/inverting like candy here because it is constexpr
    // TODO: Optimize this when these will be predominantly used in runtime
    Matrix3x3 xMat = Matrix3x3(xyzRed, xyzGreen, xyzBlue).Transpose().Inverse();
    Vector3 s = xMat * xyzWhite;
    return Matrix3x3(xyzRed   * s[0],
                     xyzGreen * s[1],
                     xyzBlue  * s[2]).Transpose();
}

MR_PF_DEF Matrix3x3 Color::GenXYZToRGB(const Primaries& p) noexcept
{
    return GenRGBToXYZ(p).Inverse();
}

MR_PF_DEF Color::Primaries Color::FindPrimaries(MRayColorSpaceEnum E)
{
    using ColorspacePrimaryList = std::array<Tuple<MRayColorSpaceEnum, Primaries>,
                                             static_cast<size_t>(MRayColorSpaceEnum::MR_END)>;

    using enum MRayColorSpaceEnum;
    constexpr ColorspacePrimaryList PRIM_LIST =
    {
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_ACES2065_1,
            Primaries
            {
                .xyRed      = Vector2(0.7347, 0.2653),
                .xyGreen    = Vector2(0, 1),
                .xyBlue     = Vector2(0.0001, -0.0770),
                .xyWhite    = ACESWhitepoint
            }
        },
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_ACES_CG,
            Primaries
            {
                .xyRed      = Vector2(0.713, 0.293),
                .xyGreen    = Vector2(0.165, 0.830),
                .xyBlue     = Vector2(0.128, 0.044),
                .xyWhite    = ACESWhitepoint
            }
        },
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_REC_709,
            Primaries
            {
                .xyRed      = Vector2(0.64, 0.33),
                .xyGreen    = Vector2(0.30, 0.60),
                .xyBlue     = Vector2(0.15, 0.06),
                .xyWhite    = D65Whitepoint
            }
        },
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_REC_2020,
            Primaries
            {
                .xyRed      = Vector2(0.708, 0.292),
                .xyGreen    = Vector2(0.170, 0.797),
                .xyBlue     = Vector2(0.131, 0.046),
                .xyWhite    = D65Whitepoint
            }
        },
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_DCI_P3,
            Primaries
            {
                .xyRed      = Vector2(0.680, 0.320),
                .xyGreen    = Vector2(0.265, 0.690),
                .xyBlue     = Vector2(0.150, 0.060),
                .xyWhite    = D65Whitepoint
            }
        },
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_ADOBE_RGB,
            Primaries
            {
                .xyRed      = Vector2(0.64, 0.33),
                .xyGreen    = Vector2(0.21, 0.71),
                .xyBlue     = Vector2(0.15, 0.06),
                .xyWhite    = D65Whitepoint
            }
        },
        Tuple<MRayColorSpaceEnum, Primaries>
        {
            MR_DEFAULT,
            Primaries
            {
                // This will create divide by zero if we set it to
                // set it to something else
                .xyRed      = Vector2(1, 0),
                .xyGreen    = Vector2(0, 1),
                .xyBlue     = Vector2(0, 0),
                .xyWhite    = Vector2(0.5)
            }
        }
    };

    auto loc = std::find_if(PRIM_LIST.cbegin(), PRIM_LIST.cend(),
                            [E](const auto& primPair)
    {
        return get<0>(primPair) == E;
    });

    // When constexpr, directly read the location.
    // Out of bounds access is UB, so this will not compile
    // since UB is an error on constexpr context.
    if(std::is_constant_evaluated())
    {
        return get<1>(*loc);
    }
    else
    {
        if(loc != PRIM_LIST.cend())
        {
            return get<1>(*loc);
        }

        #ifndef MRAY_DEVICE_CODE_PATH
            // Throw as verbose as possible
            throw MRayError("Unkown colorspace enumeration ({})! Enum should be "
                            "\"{} <= E < {}\"",
                            static_cast<uint32_t>(E), uint32_t(0),
                            static_cast<uint32_t>(MR_END));
        #elif defined MRAY_DEVICE_CODE_PATH_CUDA
            // TODO: This is CUDA only
            __trap();
        #elif defined MRAY_DEVICE_CODE_PATH_HIP
            abort();
        #endif
    }
    return Color::Primaries{};
}

template <MRayColorSpaceEnum E>
MR_PF_DEF Vector3 Color::Colorspace<E>::ToXYZ(const Vector3& rgb) const noexcept
{
    // Passing matrix to local space (GPU does not like static constexpr variables)
    constexpr auto Mat = ToXYZMatrix;
    return Mat * rgb;
}

template <MRayColorSpaceEnum E>
MR_PF_DEF Vector3 Color::Colorspace<E>::FromXYZ(const Vector3& xyz) const noexcept
{
    // Passing matrix to local space (GPU does not like static constexpr variables)
    constexpr auto Mat = FromXYZMatrix;
    return Mat * xyz;
}

template <MRayColorSpaceEnum F, MRayColorSpaceEnum T>
MR_PF_DEF Vector3 Color::ColorspaceTransfer<F, T>::Convert(const Vector3& rgb) const noexcept
{
    // Passing matrix to local space (GPU does not like static constexpr variables)
    constexpr auto Mat = RGBToRGBMatrix;
    return Mat * rgb;
}

// Constructors & Destructor
MR_PF_DEF_V Color::OpticalTransferGamma::OpticalTransferGamma(Float g) noexcept
    : gamma(g)
{}

MR_PF_DEF Vector3 Color::OpticalTransferGamma::ToLinear(const Vector3& color) const noexcept
{
    return Vector3(Math::Pow(color[0], gamma),
                   Math::Pow(color[1], gamma),
                   Math::Pow(color[2], gamma));
}

MR_PF_DEF Vector3 Color::OpticalTransferIdentity::ToLinear(const Vector3& color) const noexcept
{
    return color;
}