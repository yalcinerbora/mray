#pragma once

#include "Core/BitFunctions.h"
#include "Core/TracerI.h"
#include "Core/TypeGenFunction.h"

#include "Device/GPUSystem.h"

#include "RenderImage.h"
#include "Filters.h"

class RayPartitioner;

using TexMipBitSet = Bitset<TracerConstants::MaxTextureMipCount>;

template<class T>
using MipArray = std::array<T, TracerConstants::MaxTextureMipCount>;

struct MipGenParams
{
    TexMipBitSet validMips;
    uint16_t     mipCount;
    Vector2ui    mipZeroRes;
};

// TODO: YOLO variant is simplest implementation
// hopefully compile times do not increase as much.
// Check later.
using SurfViewVariant = Variant
<
    std::monostate,
    RWTextureView<2, Float>,
    RWTextureView<2, Vector2>,
    RWTextureView<2, Vector3>,
    RWTextureView<2, Vector4>,
    RWTextureView<2, uint8_t>,
    RWTextureView<2, Vector2uc>,
    RWTextureView<2, Vector3uc>,
    RWTextureView<2, Vector4uc>,
    RWTextureView<2, int8_t>,
    RWTextureView<2, Vector2c>,
    RWTextureView<2, Vector3c>,
    RWTextureView<2, Vector4c>,
    RWTextureView<2, uint16_t>,
    RWTextureView<2, Vector2us>,
    RWTextureView<2, Vector3us>,
    RWTextureView<2, Vector4us>,
    RWTextureView<2, int16_t>,
    RWTextureView<2, Vector2s>,
    RWTextureView<2, Vector3s>,
    RWTextureView<2, Vector4s>
>;

using SurfRefVariant = Variant
<
    std::monostate,
    RWTextureRef<2, Float>,
    RWTextureRef<2, Vector2>,
    RWTextureRef<2, Vector3>,
    RWTextureRef<2, Vector4>,
    RWTextureRef<2, uint8_t>,
    RWTextureRef<2, Vector2uc>,
    RWTextureRef<2, Vector3uc>,
    RWTextureRef<2, Vector4uc>,
    RWTextureRef<2, int8_t>,
    RWTextureRef<2, Vector2c>,
    RWTextureRef<2, Vector3c>,
    RWTextureRef<2, Vector4c>,
    RWTextureRef<2, uint16_t>,
    RWTextureRef<2, Vector2us>,
    RWTextureRef<2, Vector3us>,
    RWTextureRef<2, Vector4us>,
    RWTextureRef<2, int16_t>,
    RWTextureRef<2, Vector2s>,
    RWTextureRef<2, Vector3s>,
    RWTextureRef<2, Vector4s>
>;

class TextureFilterI
{
    public:
    virtual         ~TextureFilterI() = default;
    // Interface
    virtual void    GenerateMips(const std::vector<MipArray<SurfRefVariant>>&,
                                 const std::vector<MipGenParams>&) const = 0;
    virtual void    ReconstructionFilterRGB(// Output
                                            const SubImageSpan<3>& img,
                                            // I-O
                                            RayPartitioner& partitioner,
                                            // Input
                                            const Span<const Vector3>& dValues,
                                            const Span<const Vector2>& dImgCoords,
                                            // Constants
                                            uint32_t parallelHint,
                                            Float scalarWeightMultiplier) const = 0;
};

using TexFilterGenerator = GeneratorFuncType<TextureFilterI, const GPUSystem&, float>;

template<FilterType::E E, class FilterFunctor>
class TextureFilterT : public TextureFilterI
{
    public:
    static constexpr FilterType::E TypeName = E;

    private:
    const GPUSystem&    gpuSystem;
    Float               filterRadius;

    public:
    // Constructors & Destructor
            TextureFilterT(const GPUSystem&, Float filterRadius);
    //
    void    GenerateMips(const std::vector<MipArray<SurfRefVariant>>&,
                         const std::vector<MipGenParams>&) const override;
    void    ReconstructionFilterRGB(// Output
                                    const SubImageSpan<3>& img,
                                    // I-O
                                    RayPartitioner& partitioner,
                                    // Input
                                    const Span<const Vector3>& dValues,
                                    const Span<const Vector2>& dImgCoords,
                                    // Constants
                                    uint32_t parallelHint,
                                    Float scalarWeightMultiplier) const override;
};

extern template TextureFilterT<FilterType::BOX, BoxFilter>;
extern template TextureFilterT<FilterType::TENT, TentFilter>;
extern template TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
extern template TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;

using TextureFilterBox = TextureFilterT<FilterType::BOX, BoxFilter>;
using TextureFilterTent = TextureFilterT<FilterType::TENT, TentFilter>;
using TextureFilterGaussian = TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
using TextureFilterMitchellNetravali = TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;