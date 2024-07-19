#pragma once

#include "Device/GPUSystemForward.h"

#include "TextureCommon.h"

struct ColorConvParams
{
    TexMipBitSet        validMips;
    uint8_t             mipCount;
    MRayColorSpaceEnum  fromColorSpace;
    Float               gamma;
    Vector2ui           mipZeroRes;
};

class ColorConverter
{
    private:
    const GPUSystem& gpuSystem;

    public:
    // Constructors & Destructor
            ColorConverter(const GPUSystem&);

    void    ConvertColor(const std::vector<MipArray<SurfRefVariant>>& textures,
                         const std::vector<ColorConvParams>& params,
                         MRayColorSpaceEnum globalColorSpace) const;
};