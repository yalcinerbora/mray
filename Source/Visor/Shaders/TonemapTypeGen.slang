module TonemapTypeGen;

import Tonemap;
import ColorFunctions;

#ifdef MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB

	export struct Tonemapper		: ITonemapper	= Reinhard2002;
    export struct ColorConverterIn	: IColorConv	= AcesCGColorConv;
    export struct ColorConverterOut : IColorConv	= SRGBColorConv;

#elif defined MRAY_TONEMAP_EMPTY_ACES_CG_TO_HDR10

	export struct Tonemapper		: ITonemapper	= EmptyTonemapper;
	export struct ColorConverterIn	: IColorConv	= AcesCGColorConv;
    export struct ColorConverterOut : IColorConv	= HDR10ColorConv;

#else
	#error "Please set a tonemap type matrix"
#endif

