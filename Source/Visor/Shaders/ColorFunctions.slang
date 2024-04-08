module ColorFunctions;

// The interface assumes all of the color is in
// and *absolute* color space (no-white point etc.)
// Only non-linear space conversion happens when
// the image goes to the monitor, then correct gamma
// should be called (when swap chain is non-linear)

float3 XYZToYxy(in float3 XYZ)
{
    // Specifically component-wise divide here
    // slightly better accuracy porbably.
    float3 invSumVec = float3(1.0f) / XYZ;
    let invSum = (invSumVec.x + invSumVec.y + invSumVec.z);
    return float3(XYZ.g,
                  XYZ.r * invSum,
                  XYZ.b * invSum);
}

float3 YxyToXYZ(in float3 Yxy)
{
    // https://www.easyrgb.com/en/math.php
    // Yxy to XYZ
    float yy = (Yxy.r / Yxy.b);
    float x = Yxy.g * yy;
    float y = Yxy.r;
    float z = (1 - Yxy.g - Yxy.b) * yy;
    float3 xyz = float3(x, y, z);
    return xyz;
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
        float v = float(1.0) / gamma;
        return pow(color, gamma);
    }

    public static float3 FromLinear(in float3 color, in float gamma)
    {
        return pow(color, gamma);
    }

    public static float3 ToYxy(in float3 color)
    {
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        let mat = float3x3(0.412390798f, 0.357584357f, 0.180480793f,
                           0.212639004f, 0.715168715f, 0.072192319f,
                           0.019330820f, 0.119194776f, 0.950532079f);

        let XYZ = mul(mat, color);
        return XYZToYxy(XYZ);
    }

    public static float3 FromYxy(in float3 color)
    {
        float3 XYZ = YxyToXYZ(color);
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        // XYZ to RGB linear
        let mat = float3x3( 3.2409699419f, -1.5373831776f, -0.4986107603f,
                           -0.9692436363f,  1.8759675015f,  0.0415550574f,
                            0.0556300797f, -0.2039769589f,  1.0569715142f);
        let srgb = mul(mat, XYZ);
        return srgb;
    }
}

public struct AcesCGColorConv : IColorConv
{
    public typedef float GammaParams;

    public static float3 ToLinear(in float3 color, in float gamma)
    {
        float v = float(1.0) / gamma;
        return pow(color, gamma);
    }

    public static float3 FromLinear(in float3 color, in float gamma)
    {
        return pow(color, gamma);
    }

    public static float3 ToYxy(in float3 color)
    {
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        let mat = float3x3( 0.6624541811f, 0.1340042065f, 0.1561876870f,
                            0.2722287168f, 0.6740817658f, 0.0536895174f,
                           -0.0055746495f, 0.0040607335f, 1.0103391003f);
        let XYZ = mul(mat, color);
        // Specifically component-wise divide here
        // slightly better accuracy porbably.
        float3 invSumVec = float3(1.0f) / XYZ.x;
        let invSum = (invSumVec.x + invSumVec.y + invSumVec.z);
        return float3(XYZ.g,
                      XYZ.r * invSum,
                      XYZ.b * invSum);
    }

    public static float3 FromYxy(in float3 color)
    {
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        // Yxy to XYZ
        float yy = (color.r / color.b);
        float x = color.g * yy;
        float y = color.r;
        float z = (1 - color.g - color.b) * yy;
        float3 xyz = float3(x, y, z);

        // https://en.wikipedia.org/wiki/SRGB
        // XYZ to RGB linear
        let mat = float3x3( 1.6410233797f, -0.3248032942f, -0.2364246952f,
                           -0.6636628587f,  1.6153315917f,  0.0167563477f,
                            0.0117218943f, -0.0082844420f,  0.9883948585f);
        let srgb = mul(mat, color);
        return srgb;
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
        float Y = Yxy[0] / displayPeak;
        float yPow = pow(Y, m1);
        float top = c1 + c2 * yPow;
        float denom = 1.0f + c3 * yPow;
        float ECap = top / denom;
        Yxy[0] = pow(ECap, m2);
        return FromYxy(Yxy);
    }

    public static float3 FromLinear(in float3 color, in float displayPeak)
    {
        float3 Yxy = ToYxy(color);
        float ECap = Yxy[0];
        float ePow = pow(ECap, m2Inv);
        float top = max(ePow - c1, 0.0f);
        float denom = c2 - c3 * ePow;
        float Y = pow(top / denom, m2Inv);
        Yxy[0] = Y * displayPeak;
        return FromYxy(Yxy);
    }

    // Rec.2020 Conversion Matrices
    public static float3 ToYxy(in float3 color)
    {
        // https://www.russellcottrell.com/photo/matrixCalculator.htm
        // primaries from https://www.color.org/chardata/rgb/BT2020.xalter
        float3x3 mat = float3x3( 0.4002400f, 0.7076000f, -0.0808100f,
                                -0.2263000f, 1.1653200f,  0.0457000f,
                                 0.0000000f, 0.0000000f,  0.9182200f);
        float3 XYZ = mul(mat, color);
        return XYZToYxy(XYZ);
    }

    public static float3 FromYxy(in float3 color)
    {
        float3 XYZ = YxyToXYZ(color);
        float3x3 mat = float3x3( 1.7166512f, -0.3556708f, -0.2533663f,
                                -0.6666844f,  1.6164812f,  0.0157685f,
                                 0.0176399f, -0.0427706f, 0.9421031f);
        return mul(mat, XYZ);
    }
}