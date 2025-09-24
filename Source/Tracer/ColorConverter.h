#pragma once

#include "TextureCommon.h"

#include "Device/GPUSystemForward.h"

#include "Core/Vector.h"
#include "Core/Span.h"

class GenericTexture;
struct TracerSurfRef;

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
    //
    void ConvertColor(std::vector<MipArray<TracerSurfRef>> textures,
                      std::vector<ColorConvParams> params,
                      std::vector<GenericTexture*> bcTextures,
                      MRayColorSpaceEnum globalColorSpace) const;
    void ExtractLuminance(std::vector<Span<Float>> dLuminanceBuffers,
                          std::vector<const GenericTexture*> textures,
                          const GPUQueue& queue);
};