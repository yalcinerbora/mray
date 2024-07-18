#include "ColorConverter.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

// Constructors & Destructor
ColorConverter::ColorConverter(const GPUSystem& sys)
    : gpuSystem(sys)
{}

void ColorConverter::ConvertColor(const std::vector<MipArray<SurfRefVariant>>& textures,
                                  const std::vector<ColorConvParams>& params,
                                  MRayColorSpaceEnum globalColorSpace) const
{

}