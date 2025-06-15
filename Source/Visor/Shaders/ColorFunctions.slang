module ColorFunctions;

// The [To/From]Yxy interface assumes all of the color is in
// a linear D65 color space.
// [To/From]Linear is responsible for EOTF conversion.
// Currently only ACES-cg has non D65 whitepoint,
// so we include Bradford chromacity adaptation
// matrix to convert the whitepoints.
// Implementation is from here
// https://www.color.org/specification/icc1v43_2010-12.pdf
float3 XYZToYxy(in float3 xyz)
{
    let invSum = float(1) / (xyz.x + xyz.y + xyz.z);
    // This has slightly better precision maybe?
    let x = float(1) - (xyz.y + xyz.z) * invSum;
    let y = float(1) - (xyz.x + xyz.z) * invSum;
    return float3(xyz.y, x, y);
}

float3 YxyToXYZ(in float3 Yxy)
{
    // https://www.easyrgb.com/en/math.php
    // Yxy to XYZ
    float yy = (Yxy.r / Yxy.b);
    float x = Yxy.g * yy;
    float y = Yxy.r;
    float z = (float(1) - Yxy.g - Yxy.b) * yy;
    float3 xyz = float3(x, y, z);
    return xyz;
}

// TODO: Hand constexpr these for default whitepoints.
static const float3x3 BradfordMatrix = float3x3
(
     0.8951f,  0.2664f, -0.1614f,
    -0.7502f,  1.7135f,  0.0367f,
     0.0389f, -0.0685f,  1.0296f
);

static const float3x3 InvBradfordMatrix = float3x3
(
     0.9869929f, -0.1470543f, 0.1599627f,
     0.4323053f,  0.5183603f, 0.0492912f,
    -0.0085287f,  0.0400428f, 0.9684867f
);

// https://en.wikipedia.org/wiki/Standard_illuminant
static const float3 D65WhiteXYZ = float3(0.95047f, 1.0f, 1.08883f);
// Hand calculated from the xy whitepoint of D60
static const float3 D60WhiteXYZ = float3(0.95265f, 1.0f, 1.00883f);

float3 ChangeWhitePoint(in float3 outWhitePoint,
                        in float3 inWhitePoint,
                        in float3 xyz)
{
    float3 coneRespIn = mul(BradfordMatrix, inWhitePoint);
    float3 coneRespOut = mul(BradfordMatrix, outWhitePoint);
    float3 d = coneRespOut / coneRespIn;
    float3x3 scaleMatrix = float3x3(d[0], 0,    0,
                                    0,    d[1], 0,
                                    0,    0,    d[2]);
    float3x3 matrix = mul(InvBradfordMatrix, mul(scaleMatrix, BradfordMatrix));
    return mul(matrix, xyz);
}

public interface IColorConv
{
    public associatedtype GammaParams;

    static float3 ToLinear(in float3 color, in GammaParams gammaParams);
    static float3 FromLinear(in float3 color, in GammaParams gammaParams);
    static float3 ToYxy(in float3 color);
    static float3 FromYxy(in float3 color);
}

public struct SRGBColorConv : IColorConv
{
    public typedef float GammaParams;

    public static float3 ToLinear(in float3 color, in float gamma)
    {
        return pow(color, gamma);
    }

    public static float3 FromLinear(in float3 color, in float gamma)
    {
        float v = float(1) / gamma;
        return pow(color, v);
    }

    public static float3 ToYxy(in float3 color)
    {
        // https://observablehq.com/@danburzo/color-matrix-calculator
        let mat = float3x3
        (
            0.4123907992659593f, 0.3575843393838780f, 0.1804807884018343f,
            0.2126390058715102f, 0.7151686787677560f, 0.0721923153607337f,
            0.0193308187155918f, 0.1191947797946260f, 0.9505321522496607f
        );
        let xyz = mul(mat, color);
        return XYZToYxy(xyz);
    }

    public static float3 FromYxy(in float3 color)
    {
        let xyz = YxyToXYZ(color);
        // https://observablehq.com/@danburzo/color-matrix-calculator
        // XYZ to RGB linear
        let mat = float3x3
        (
           3.2409699419045226f, -1.5373831775700939f, -0.4986107602930034f,
          -0.9692436362808796f,  1.8759675015077204f,  0.0415550574071756f,
           0.0556300796969936f, -0.2039769588889765f,  1.0569715142428784f
        );
        let srgb = mul(mat, xyz);
        return srgb;
    }
}

public struct AcesCGColorConv : IColorConv
{
    public typedef float GammaParams;

    public static float3 ToLinear(in float3 color, in float gamma)
    {
        let v = float(1) / gamma;
        return pow(color, gamma);
    }

    public static float3 FromLinear(in float3 color, in float gamma)
    {
        return pow(color, gamma);
    }

    public static float3 ToYxy(in float3 color)
    {
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        let mat = float3x3
        (
             0.6624541811f, 0.1340042065f, 0.1561876870f,
             0.2722287168f, 0.6740817658f, 0.0536895174f,
            -0.0055746495f, 0.0040607335f, 1.0103391003f
        );
        float3 xyz = mul(mat, color);
        // Input color is at D65, change to D60
        xyz = ChangeWhitePoint(D65WhiteXYZ, D60WhiteXYZ, xyz);

        return XYZToYxy(xyz);
    }

    public static float3 FromYxy(in float3 color)
    {
        float3 xyz = YxyToXYZ(color);
        // Input color is at D60, change to D65
        xyz = ChangeWhitePoint(D60WhiteXYZ, D65WhiteXYZ, xyz);
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        // XYZ to RGB linear
        let mat = float3x3
        (
             1.6410233797f, -0.3248032942f, -0.2364246952f,
            -0.6636628587f,  1.6153315917f,  0.0167563477f,
             0.0117218943f, -0.0082844420f,  0.9883948585f
        );
        return mul(mat, xyz);
    }
}

public struct HDR10ColorConv : IColorConv
{
    public typedef float GammaParams;

    // SMPTE ST 2084 transfer functions
    // Adjustable parameter is the diplay's peak brightness.
    // https://en.wikipedia.org/wiki/Perceptual_quantizer
    private static const float c1 = 0.8359375f;
    private static const float c2 = 18.8515625f;
    private static const float c3 = 18.6875f;
    private static const float m1 = 0.1593017578125f;
    private static const float m1Inv = 1.0f / m1;
    private static const float m2 = 78.84375f;
    private static const float m2Inv = 1.0f / m2;

    public static float3 ToLinear(in float3 color, in float displayPeak)
    {
        float3 Yxy = ToYxy(color);
        // TODO: Should we clamp or normalize?
        // currently normalizing
        float eCap = Yxy[0];
        float ePow = pow(eCap, m2Inv);
        float top = max(ePow - c1, 0.0f);
        float denom = c2 - c3 * ePow;
        float Y = pow(top / denom, m1Inv);
        Yxy[0] = Y;
        return FromYxy(Yxy);
    }

    public static float3 FromLinear(in float3 color, in float displayPeak)
    {
        float3 Yxy = ToYxy(color);
        float yPow = pow(Yxy[0], m1);
        float top = c1 + c2 * yPow;
        float denom = 1.0f + c3 * yPow;
        float eCap = top / denom;
        Yxy[0] = pow(eCap, m2);
        return FromYxy(Yxy);
    }

    // Rec.2020 Conversion Matrices
    public static float3 ToYxy(in float3 color)
    {
        // https://observablehq.com/@danburzo/color-matrix-calculator
        float3x3 mat = float3x3
        (
            0.63695804830f, 0.14461690350f, 0.16888097516f,
            0.26270021201f, 0.67799807152f, 0.05930171647f,
            0.00000000000f, 0.02807269305f, 1.060985057711f
        );
        float3 xyz = mul(mat, color);
        return XYZToYxy(xyz);
    }

    public static float3 FromYxy(in float3 color)
    {
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        float3 XYZ = YxyToXYZ(color);
        float3x3 mat = float3x3
        (
             1.7166511880f, -0.3556707838f, -0.2533662814f,
            -0.6666843518f,  1.6164812366f,  0.0157685458f,
             0.0176398574f, -0.0427706133f,  0.9421031212f
        );
        return mul(mat, XYZ);
    }
}