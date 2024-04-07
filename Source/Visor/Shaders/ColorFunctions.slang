module ColorFunctions;

// The interface assumes all of the color is in
// and *absolute* color space (no-white point etc.)
// Only non-linear space conversion happens when
// the image goes to the monitor, then correct gamma
// should be called (when swap chain is non-linear)
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
        // https://www.easyrgb.com/en/math.php
        // Yxy to XYZ
        float yy = (color.r / color.b);
        float x = color.g * yy;
        float y = color.r;
        float z = (1 - color.g - color.b) * yy;
        float3 xyz = float3(x, y, z);

        // https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
        // XYZ to RGB linear
        let mat = float3x3( 3.2409699419f, -1.5373831776f, -0.4986107603f,
                           -0.9692436363f,  1.8759675015f,  0.0415550574f,
                            0.0556300797f, -0.2039769589f,  1.0569715142f);
        let srgb = mul(mat, color);
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