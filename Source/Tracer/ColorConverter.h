#pragma once

#include "Device/GPUSystemForward.h"

#include "TextureCommon.h"

class GenericTexture;

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

    void    ConvertColor(std::vector<MipArray<SurfRefVariant>> textures,
                         std::vector<ColorConvParams> params,
                         std::vector<GenericTexture*> bcTextures,
                         MRayColorSpaceEnum globalColorSpace) const;
};