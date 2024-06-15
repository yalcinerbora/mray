#pragma once

#include "Core/System.h"
#include "Core/Flag.h"
#include "Core/Types.h"
#include "Core/Expected.h"

#ifdef MRAY_GFGCONVERTER_SHARED_EXPORT
#define MRAY_GFGCONVERTER_ENTRYPOINT MRAY_DLL_EXPORT
#else
#define MRAY_GFGCONVERTER_ENTRYPOINT MRAY_DLL_IMPORT
#endif

//
namespace MRayConvert
{

enum class ConvFlagEnum
{
    PACK_GFG,
    FAIL_ON_OVERWRITE,
    NORMAL_AS_QUATERNION,
    END
};
using ConversionFlags = Flag<ConvFlagEnum>;

// GFG Conversion
MRAY_GFGCONVERTER_ENTRYPOINT
Expected<double> ConvertMeshesToGFG(const std::string& outFileName,
								    const std::string& inFileName,
                                    ConversionFlags flags);

}