module TonemapTypeGen;

import Tonemap;
import TonemapFunctions;
import ColorFunctions;
//import ColorFunctions;

#define MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB

#ifdef MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB
	export struct Tonemapper : ITonemapper = Reinhard2002;
	export struct ColorConverterIn : IColorConv = SRGBColorConv;
	export struct ColorConverterOut : IColorConv = AcesCGColorConv;
#else
	#error "Please set a tonemap matrix"
#endif

