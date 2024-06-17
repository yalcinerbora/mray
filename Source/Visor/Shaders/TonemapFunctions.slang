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
                                 in float maxLuminance,
                                 in float avgLuminance);
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
                                 in ColorConverterOut.GammaParams gammaParams,
                                 in TMParamsReinhard tmParams,
                                 in float maxLuminance,
                                 in float avgLuminance)
    {
        // In is assumed to be in linear space
        float3 hdrYxy = inConv.ToYxy(hdrColor);
        float L = hdrYxy[0];

        // Equation 2
        if(tmParams.doKeyAdjust)
            L *= (tmParams.key / avgLuminance);

        // Equation 4
        float white = maxLuminance * tmParams.burnRatio;
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
                                 in float maxLuminance,
                                 in float avgLuminance)
    {
        // In is assumed to be in linear space
        float3 hdrYxy = inConv.ToYxy(hdrColor);
        float3 sdrColor = outConv.FromYxy(hdrYxy);
        float3 finalColor = outConv.FromLinear(sdrColor, gammaParams);
        return finalColor;
    }
}
