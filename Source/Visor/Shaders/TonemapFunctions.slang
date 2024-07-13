implementing Tonemap;
import ColorFunctions;

public extern struct ColorConverterIn : IColorConv;
public extern struct ColorConverterOut : IColorConv;

public interface ITonemapper
{
    public associatedtype TMParams;
    public static float3 ToneMap(in float3 hdrColor,
                                 in ColorConverterIn,
                                 in ColorConverterOut,
                                 in ColorConverterOut.GammaParams gammaParams,
                                 in TMParams tmParams,
                                 in float2 minMaxLuminance,
                                 in float2 avgLuminance);
}

public struct Reinhard2002 : ITonemapper
{
    public struct TMParamsReinhard
    {
        bool doKeyAdjust;
        float burnRatio;
        float key;
    }

    public typedef TMParamsReinhard TMParams;

    public static float3 ToneMap(in float3 hdrColor,
                                 in ColorConverterIn inConv,
                                 in ColorConverterOut outConv,
                                 in ColorConverterOut::GammaParams gammaParams,
                                 in TMParamsReinhard tmParams,
                                 in float2 minMaxLuminance,
                                 in float2 avgLuminance)
    {
        // Image is assumed to be in linear space
        float3 hdrYxy = inConv.ToYxy(hdrColor);
        float L = hdrYxy[0];
        // Equation 2
        if (tmParams.doKeyAdjust)
        {
            L /= avgLuminance[0];
            L *= tmParams.key;
        }

        // Equation 4
        // To incorporate a burn ratio, we lerp between
        // 2x maxLuminance <-> minLuminance
        // we do logarithmic interpolation to make
        // GUI slider to be smooth when there is a sun in the
        // frame
        float white = lerp(log(minMaxLuminance[0]),
                           log(2.0f * minMaxLuminance[1]),
                           tmParams.burnRatio * 0.5f);
        white = exp(white);

        float whiteSqr = white * white;
        float sdrL = L * (1 + (L / whiteSqr)) / (1 + L);
        hdrYxy[0] = sdrL;

        float3 sdrColor = outConv.FromYxy(hdrYxy);
        float3 finalColor = outConv.FromLinear(sdrColor, gammaParams);
        return finalColor;
    }
}

public struct EmptyTonemapper : ITonemapper
{
    public typedef uint TMParams;

    public static float3 ToneMap(in float3 hdrColor,
                                 in ColorConverterIn inConv,
                                 in ColorConverterOut outConv,
                                 in ColorConverterOut::GammaParams gammaParams,
                                 in uint,
                                 in float2 minMaxLuminance,
                                 in float2)
    {
        // In is assumed to be in linear space
        float3 hdrYxy = inConv.ToYxy(hdrColor);
        // Straight up divide here
        // HDR10 will handle it? (TODO: Check this?)
        hdrYxy[0] /= minMaxLuminance[1];
        //
        float3 sdrColor = outConv.FromYxy(hdrYxy);
        float3 finalColor = outConv.FromLinear(sdrColor, gammaParams);
        return finalColor;
    }
}
