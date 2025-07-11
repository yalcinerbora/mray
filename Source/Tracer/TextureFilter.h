#pragma once

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

// TODO: Move these functions to somewhere proper
void SetImagePixels(// Output
                    const ImageSpan& img,
                    // Input
                    const Span<const Spectrum>& dValues,
                    const Span<const Float>& dFilterWeights,
                    const Span<const ImageCoordinate>& dImgCoords,
                    // Constants
                    Float scalarWeightMultiplier,
                    const GPUQueue& queue);

void SetImagePixelsIndirect(// Output
                            const ImageSpan& img,
                            // Input
                            const Span<const RayIndex>& dIndices,
                            const Span<const Spectrum>& dValues,
                            const Span<const Float>& dFilterWeights,
                            const Span<const ImageCoordinate>& dImgCoords,
                            // Constants
                            Float scalarWeightMultiplier,
                            const GPUQueue& queue);

void SetImagePixelsIndirectAtomic(// Output
                                  const ImageSpan& img,
                                  // Input
                                  const Span<const RayIndex>& dIndices,
                                  const Span<const Spectrum>& dValues,
                                  const Span<const Float>& dFilterWeights,
                                  const Span<const ImageCoordinate>& dImgCoords,
                                  // Constants
                                  Float scalarWeightMultiplier,
                                  const GPUQueue& queue);
class TextureFilterI
{
    public:
    virtual         ~TextureFilterI() = default;
    // Interface
    virtual void    GenerateMips(const std::vector<MipArray<SurfRefVariant>>&,
                                 const std::vector<MipGenParams>&) const = 0;
    virtual void    ClampImageFromBuffer(// Output
                                         const SurfRefVariant& surf,
                                         // Input
                                         const Span<const Byte>& dDataBuffer,
                                         // Constants
                                         const Vector2ui& surfImageDims,
                                         const Vector2ui& bufferImageDims,
                                         const GPUQueue& queue) const = 0;
    virtual void    ReconstructionFilterRGB(// Output
                                            const ImageSpan& img,
                                            // I-O
                                            RayPartitioner& partitioner,
                                            // Input
                                            const Span<const Spectrum>& dValues,
                                            const Span<const ImageCoordinate>& dImgCoords,
                                            // Constants
                                            uint32_t parallelHint,
                                            Float scalarWeightMultiplier,
                                            const GPUQueue& queue) const = 0;

    virtual void    ReconstructionFilterAtomicRGB(// Output
                                                  const ImageSpan& img,
                                                  // Input
                                                  const Span<const Spectrum>& dValues,
                                                  const Span<const ImageCoordinate>& dImgCoords,
                                                  // Constants
                                                  Float scalarWeightMultiplier,
                                                  const GPUQueue& queue) const = 0;
    virtual Vector2ui FilterExtent() const = 0;
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
    void    ClampImageFromBuffer(// Output
                                 const SurfRefVariant& surf,
                                 // Input
                                 const Span<const Byte>& dDataBuffer,
                                 // Constants
                                 const Vector2ui& surfImageDims,
                                 const Vector2ui& bufferImageDims,
                                 const GPUQueue& queue) const override;

    void    ReconstructionFilterRGB(// Output
                                    const ImageSpan& img,
                                    // I-O
                                    RayPartitioner& partitioner,
                                    // Input
                                    const Span<const Spectrum>& dValues,
                                    const Span<const ImageCoordinate>& dImgCoords,
                                    // Constants
                                    uint32_t parallelHint,
                                    Float scalarWeightMultiplier,
                                    const GPUQueue& queue) const override;

    void    ReconstructionFilterAtomicRGB(// Output
                                          const ImageSpan& img,
                                          // Input
                                          const Span<const Spectrum>& dValues,
                                          const Span<const ImageCoordinate>& dImgCoords,
                                          // Constants
                                          Float scalarWeightMultiplier,
                                          const GPUQueue& queue) const override;

    Vector2ui FilterExtent() const override;
};

extern template class TextureFilterT<FilterType::BOX, BoxFilter>;
extern template class TextureFilterT<FilterType::TENT, TentFilter>;
extern template class TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
extern template class TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;

using TextureFilterBox = TextureFilterT<FilterType::BOX, BoxFilter>;
using TextureFilterTent = TextureFilterT<FilterType::TENT, TentFilter>;
using TextureFilterGaussian = TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
using TextureFilterMitchellNetravali = TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;