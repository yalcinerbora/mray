module TonmapFunctions;
import ColorFunctions;

public extern struct ColorConverterIn : IColorConv;
public extern struct ColorConverterOut : IColorConv;

public interface ITonemapper
{
    public associatedtype TMParams;
    public static float3 ToneMap(in float3 hdrColor,
                                 in ColorConverterIn,
                                 in ColorConverterOut,
                                 in ColorConverterOut::GammaParams params,
                                 in float maxLuminance,
                                 in float avgLuminance);
}

public struct Reinhard2002 : ITonemapper
{
    public struct TMParamsReinhard
    {
        bool doToneMap;
        bool doGamma;
        bool doKeyAdjust;
        float gammaValue;
        float burnRatio;
        float key;
    }

    public typedef TMParamsReinhard TMParams;

    public static float3 ToneMap(in float3 hdrColor,
                                 in ColorConverterIn,
                                 in ColorConverterOut,
                                 in ColorConverterOut::GammaParams params,
                                 in float maxLuminance,
                                 in float avgLuminance)
    {
        return float3(0.0f);
    }
}

