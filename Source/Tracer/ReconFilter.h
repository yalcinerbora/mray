#pragma once

#include "Core/BitFunctions.h"
#include "Core/TracerI.h"

#include "Device/GPUSystem.h"

#include "RenderImage.h"

class RayPartitioner;

using TexMipBitSet = Bitset<TracerConstants::MaxTextureMipCount>;

template<class T>
using MipArray = std::array<T, TracerConstants::MaxTextureMipCount>;

// TODO: YOLO variant is simplest implementation
// hopefully compile times do not increase as much.
// Check later.
using SurfViewVariant = Variant
<
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

struct GenMipInput
{
};

class ReconstructionFilterI
{
    public:
    virtual         ~ReconstructionFilterI() = default;
    // Interface
    virtual void    GenerateMips(const std::vector<MipArray<SurfRefVariant>>&,
                                 uint32_t seed) const = 0;

    //virtual uint32_t    FilterGridSize() const = 0;
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

class ReconstructionFilterBox : public ReconstructionFilterI
{
    private:
    const GPUSystem&    gpuSystem;
    Float               filterRadius;

    public:
    static std::string_view TypeName();
    // Constructors & Destructor
            ReconstructionFilterBox(const GPUSystem&,
                                    Float filterRadius);
    //
    void    GenerateMips(const std::vector<MipArray<SurfRefVariant>>&,
                         uint32_t seed) const override;
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

class ReconstructionFilterMitchell : public ReconstructionFilterI
{
    private:
    const GPUSystem&    gpuSystem;
    Float               filterRadius;
    Float               b;
    Float               c;

    public:
    static std::string_view TypeName();
    // Constructors & Destructor
    ReconstructionFilterMitchell(const GPUSystem&,
                                 Float filterRadius,
                                 Float b, Float c);
    //
    void    GenerateMips(const std::vector<MipArray<SurfRefVariant>>&,
                         uint32_t seed) const override;
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