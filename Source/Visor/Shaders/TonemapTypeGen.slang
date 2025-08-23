module TonemapTypeGen;

import Tonemap;
import ColorFunctions;

#ifdef MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB

    export struct ColorConverterIn  : IColorConv    = AcesCGColorConv;
    export struct ColorConverterOut : IColorConv    = SRGBColorConv;
    export struct Tonemapper        : ITonemapper   = Reinhard2002;

#elif defined MRAY_TONEMAP_EMPTY_ACES_CG_TO_HDR10

    export struct ColorConverterIn  : IColorConv    = AcesCGColorConv;
    export struct ColorConverterOut : IColorConv    = HDR10ColorConv;
    export struct Tonemapper        : ITonemapper   = EmptyTonemapper;

#else
    #error "Please set a tonemap type matrix"
#endif

