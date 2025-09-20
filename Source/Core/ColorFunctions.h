#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Types.h"

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

    template<std::floating_point F>
    MR_PF_DECL Vector<3, F> XYZToCIELab(const Vector<3, F>& xyz,
                                        const Vector<3, F>& whitepoint) noexcept;

    template<std::floating_point F>
    MR_PF_DECL Vector<3, F> CIELabToXYZ(const Vector<3, F>& cieLab,
                                        const Vector<3, F>& whitepoint) noexcept;

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
    inline constexpr Vector2 D50Whitepoint = Vector2(0.34567, 0.35850);
    inline constexpr Vector2 D55Whitepoint = Vector2(0.33242, 0.34743);
    // These are different
    // https://docs.acescentral.com/tb/white-point#comparison-of-the-aces-white-point-and-cie-d60
    inline constexpr Vector2 D60Whitepoint = Vector2(0.32169, 0.33780);
    inline constexpr Vector2 ACESWhitepoint = Vector2(0.32168, 0.33767);
    inline constexpr Vector2 D65Whitepoint = Vector2(0.31272, 0.32903);
    inline constexpr Vector2 D75Whitepoint = Vector2(0.29902, 0.31485);

    //
    // Data from here: https://www.rit.edu/science/munsell-color-science-lab-educational-resources
    // Now cie data[1] does not match this, so one of which is normalized maybe?
    //
    // [1]: https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
    inline constexpr Vector2ui CIE_1931_RANGE   = Vector2ui(360, 831);
    inline constexpr Float CIE_1931_DELTA       = Float(1);
    inline constexpr uint32_t CIE_1931_N        = CIE_1931_RANGE[1] - CIE_1931_RANGE[0];
    // Data is on the CPP to reduce parsing maybe?
    extern const std::array<Vector3, CIE_1931_N> CIE_1931_XYZ;
    // Generated using this: https://draftdocs.acescentral.com/white-point/
    // repo: https://github.com/ampas/aces-docs/blob/main/python/TB-2018-001/aces_wp.py
    extern const std::array<Float, CIE_1931_N> D60_SPD;
    extern const std::array<Float, CIE_1931_N> D65_SPD;
    extern const std::array<Float, CIE_1931_N> D75_SPD;
    // TODO: Where are these factors comes from?
    // These are from here: https://github.com/mitsuba-renderer/rgb2spec
    // (I've checked its not PWC integral, PWL(trapz) integral.
    // So it must be something else?)
    inline constexpr Float D60_SPD_NORM_FACTOR = Float(10536.3);
    inline constexpr Float D65_SPD_NORM_FACTOR = Float(10566.864);
    // TODO: Since I do not know how are these calculated and paper / refimpl
    // do not have D75 whitepoint, setting it to D65 these are closeby so
    // it should be fine I hope...
    inline constexpr Float D75_SPD_NORM_FACTOR = D65_SPD_NORM_FACTOR;

    // Ad-hoc fitted gaussian of full CIE 1931 Observer
    // https://www.desmos.com/calculator/zepnypxnmd
    inline constexpr Vector2 CIE_1931_MIS = Vector2(0.384615384615, 0.615384615385);
    inline constexpr Vector2 CIE_1931_GAUSS_SIGMA = Vector2(25, 48);
    inline constexpr Vector2 CIE_1931_GAUSS_MU = Vector2(452, 576);
    static_assert(CIE_1931_MIS.Sum() == Float(1), "Wrong MIS Ratio for CIE_1931 Gaussians!");

    const std::array<Float, CIE_1931_N>&
    SelectIlluminantSPD(MRayColorSpaceEnum);

    //
    Float       SelectIlluminantSPDNormFactor(MRayColorSpaceEnum e);
    Matrix3x3   SelectRGBToXYZMatrix(MRayColorSpaceEnum e);

    static constexpr std::string_view LUT_FILE_CC = "MR_SPECTRA";
    static constexpr std::string_view LUT_FILE_EXT = ".mrspectra";
    // TODO: This should come from CMake, since build system uses it
    // as well.
    static constexpr std::string_view LUT_FOLDER_NAME = "SpectraLUT";

    MR_PF_DECL Primaries FindPrimaries(MRayColorSpaceEnum);

    // Colorspace guarantees whitepoints match
    // by enforcing every colorspace to convert to D65
    // (Only for ACES spaces)
    template <MRayColorSpaceEnum E>
    class Colorspace
    {
        public:
        static constexpr Primaries Prims = FindPrimaries(E);

        static constexpr Matrix3x3 ToXYZMatrix = (E == MRayColorSpaceEnum::MR_DEFAULT)
            ? Matrix3x3::Identity()
            : (GenWhitepointMatrix(Prims.xyWhite, D65Whitepoint) * GenRGBToXYZ(Prims));

        static constexpr Matrix3x3 FromXYZMatrix = ToXYZMatrix.Inverse();

        static constexpr Vector3 WhiteXYZ = GenWhitepointXYZ(Prims.xyWhite);

        static constexpr auto Enum = E;

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

template<std::floating_point F>
MR_PF_DEF Vector<3, F> Color::XYZToCIELab(const Vector<3, F>& xyz,
                                          const Vector<3, F>& whitepoint) noexcept
{
    // https://en.wikipedia.org/wiki/CIELAB_color_space
    auto Func = [](F t)
    {
        constexpr F D = F(6) / F(29);
        constexpr F DCube = D * D * D;
        if(t > DCube)
            return Math::Cbrt(t);
        else
        {
            constexpr F Case2Factor = F(1) / (D * D * F(3));
            constexpr F C = F(4) / F(29);
            return t * Case2Factor + C;
        }
    };

    F xN = Func(xyz[0] / whitepoint[0]);
    F yN = Func(xyz[1] / whitepoint[1]);
    F zN = Func(xyz[2] / whitepoint[2]);
    F l = F(116) * yN - F(16);
    F a = F(500) * (xN - yN);
    F b = F(200) * (yN - zN);
    return Vector<3, F>(l, a, b);
}

template<std::floating_point F>
MR_PF_DEF Vector<3, F> Color::CIELabToXYZ(const Vector<3, F>& cieLab,
                                          const Vector<3, F>& whitepoint) noexcept
{
    // https://en.wikipedia.org/wiki/CIELAB_color_space
    auto FuncInv = [](F t)
    {
        constexpr F D = F(6) / F(29);
        if(t > D)
            return t * t * t;
        else
        {
            constexpr F Case2Factor = F(3) * D * D;
            constexpr F C = F(4) / F(29);
            return t - C * Case2Factor;
        }
    };
    constexpr F LFactor = F(1) / F(116);
    constexpr F AFactor = F(1) / F(500);
    constexpr F BFactor = F(1) / F(200);

    F l = (cieLab[0] + F(16)) * LFactor;
    F a = cieLab[1] * AFactor;
    F b = cieLab[2] * BFactor;
    //
    F x = FuncInv(l + a)   * whitepoint[0];
    F y = FuncInv(l)       * whitepoint[1];
    F z = FuncInv(l - b)   * whitepoint[2];
    return Vector<3, F>(x, y, z);
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
        HybridTerminateOrTrap("Unkown colorspace enumeration at "
                              "\"Color::FindPrimaries\"!");
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