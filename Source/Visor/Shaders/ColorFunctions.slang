module ColorFunctions;

// The interface assumes all of the color is in
// and *absolute* color space (no-white point etc.)
// Only non-linear space conversion happens when
// the image goes to the monitor, then correct gamma
// should be called (when swap chain is non-linear)

float3 XYZToYxy(in float3 XYZ)
{
    let invSum = 1 / (XYZ.x + XYZ.y + XYZ.z);
    // this has slightly better precision maybe?
    let x = 1 - (XYZ.y + XYZ.z) * invSum;
    let y = 1 - (XYZ.x + XYZ.z) * invSum;
    return float3(XYZ.y, x, y);
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

        let xyz = mul(mat, color);
        return XYZToYxy(xyz);
    }

    public static float3 FromYxy(in float3 color)
    {
        float3 xyz = YxyToXYZ(color);
        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        // XYZ to RGB linear
        let mat = float3x3( 3.2409699419f, -1.5373831776f, -0.4986107603f,
                           -0.9692436363f,  1.8759675015f,  0.0415550574f,
                            0.0556300797f, -0.2039769589f,  1.0569715142f);
        let srgb = mul(mat, xyz);
        // return srgb;
        return xyz;
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
        let xyz = mul(mat, color);
        return XYZToYxy(xyz);
    }

    public static float3 FromYxy(in float3 color)
    {
        float3 xyz = YxyToXYZ(color);
        // https://en.wikipedia.org/wiki/SRGB
        // XYZ to RGB linear
        let mat = float3x3( 1.6410233797f, -0.3248032942f, -0.2364246952f,
                           -0.6636628587f,  1.6153315917f,  0.0167563477f,
                            0.0117218943f, -0.0082844420f,  0.9883948585f);
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

    // Rec.2020 Conversion Matrices
    public static float3 ToYxy(in float3 color)
    {
        // https://observablehq.com/@danburzo/color-matrix-calculator
        float3x3 mat = float3x3(0.6369535f, 0.1446192f, 0.1688559f,
                                0.2626983f, 0.6780088f, 0.0592920f,
                                0.0000000f, 0.0280731f, 1.0608272f);
        float3 xyz = mul(mat, color);
        return XYZToYxy(xyz);
    }

    public static float3 FromYxy(in float3 color)
    {
        // https://observablehq.com/@danburzo/color-matrix-calculator
        float3 XYZ = YxyToXYZ(color);
        float3x3 mat = float3x3( 1.7166634f, -0.3556733f, -0.2533681f,
                                -0.6666738f,  1.6164557f,  0.01576823f,
                                 0.0176425f, -0.0427770f,  0.9422433f);
        return mul(mat, XYZ);
    }
}