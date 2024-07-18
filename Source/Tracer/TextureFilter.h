#pragma once

#include "Core/BitFunctions.h"
#include "Core/TracerI.h"
#include "Core/TypeGenFunction.h"

#include "Device/GPUSystemForward.h"

#include "TextureCommon.h"
#include "RenderImage.h"
#include "Filters.h"

class RayPartitioner;

struct alignas(Vector4ui) MipGenParams
{
    TexMipBitSet validMips;
    uint16_t     mipCount;
    Vector2ui    mipZeroRes;
};



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