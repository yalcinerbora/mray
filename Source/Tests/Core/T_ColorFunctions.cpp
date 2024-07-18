#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>


#include "Core/Vector.h"
#include "Core/Quaternion.h"
#include "Core/ColorFunctions.h"

#include "GTestWrappers.h"

TEST(ColorFunctionsTest, ACES)
{
    using namespace Color;
    using enum MRayColorSpaceEnum;
    using AcesAP0Colorspace = Colorspace<MR_ACES2065_1>;
    using AcesCGColorspace = Colorspace<MR_ACES_CG>;

    // From
    // https://docs.acescentral.com/specifications/acescg/#color-space
    // TRA_1
    static constexpr Matrix3x3 TRA_1_Ref = Matrix3x3
    (
         1.4514393161, -0.2365107469, -0.2149285693,
        -0.0765537734,  1.1762296998, -0.0996759264,
         0.0083161484, -0.0060324498,  0.9977163014
    );
    static constexpr Matrix3x3 TRA_2_Ref = Matrix3x3
    (
         0.6954522414f, 0.1406786965f, 0.1638690622f,
         0.0447945634f, 0.8596711185f, 0.0955343182f,
        -0.0055258826f, 0.0040252103f, 1.0015006723f
    );
    // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    static constexpr Matrix3x3 ToXYZ_AcesCG_Ref = Matrix3x3
    (
        0.6624541811f, 0.1340042065f, 0.1561876870f,
        0.2722287168f, 0.6740817658f, 0.0536895174f,
        -0.0055746495f, 0.0040607335f, 1.0103391003f
    );
    static constexpr Matrix3x3 ToXYZ_AcesAP0_Ref = Matrix3x3
    (
        0.9525523959f, 0.0000000000f,  0.0000936786f,
        0.3439664498f, 0.7281660966f, -0.0721325464f,
        0.0000000000f, 0.0000000000f,  1.0088251844f
    );

    // Generated values
    static constexpr Matrix3x3 ToXYZ_AcesCG = GenRGBToXYZ(FindPrimaries(MR_ACES_CG));
    static constexpr Matrix3x3 ToXYZ_AcesAP0 = GenRGBToXYZ(FindPrimaries(MR_ACES2065_1));
    // We indirectly test Chromaticity transform here
    // These convert from ~D60 (ACES White) to D65
    static constexpr Matrix3x3 TRA_1 = (AcesCGColorspace::FromXYZMatrix *
                                        AcesAP0Colorspace::ToXYZMatrix);
    static constexpr Matrix3x3 TRA_2 = (AcesAP0Colorspace::FromXYZMatrix *
                                        AcesCGColorspace::ToXYZMatrix);
    // We can static assert here since the codepaths are the same
    // we are doing this to check if we do wrong multiplication order etc.
    static_assert(ColorspaceTransfer<MR_ACES2065_1, MR_ACES_CG>::RGBToRGBMatrix == TRA_1);
    static_assert(ColorspaceTransfer<MR_ACES_CG, MR_ACES2065_1>::RGBToRGBMatrix == TRA_2);
    // Tightyly bounding these by hand
    static constexpr auto TestEpsilon = Float(2.4e-7);
    EXPECT_NEAR_MRAY(TRA_1, TRA_1_Ref, TestEpsilon);
    EXPECT_NEAR_MRAY(TRA_2, TRA_2_Ref, TestEpsilon);
    EXPECT_NEAR_MRAY(ToXYZ_AcesCG, ToXYZ_AcesCG_Ref, TestEpsilon);
    EXPECT_NEAR_MRAY(ToXYZ_AcesAP0, ToXYZ_AcesAP0_Ref, TestEpsilon);
}
