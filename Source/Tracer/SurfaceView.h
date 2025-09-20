#pragma once

#include "Device/GPUTexture.h"
#include "Device/GPUTextureView.h"

#include "Core/Variant.h"

// TODO: YOLO variant is simplest implementation
// hopefully compile times do not increase as much.
// Check later.
using SurfViewVariant = Variant
<
    std::monostate,
    // Float
    RWTextureView<2, Float>,
    RWTextureView<2, Vector2>,
    RWTextureView<2, Vector3>,
    RWTextureView<2, Vector4>,
    // 8-bit Unsigned
    RWTextureView<2, uint8_t>,
    RWTextureView<2, Vector2uc>,
    RWTextureView<2, Vector3uc>,
    RWTextureView<2, Vector4uc>,
    // 8-bit Signed
    RWTextureView<2, int8_t>,
    RWTextureView<2, Vector2c>,
    RWTextureView<2, Vector3c>,
    RWTextureView<2, Vector4c>,
    // 16-bit Unsigned
    RWTextureView<2, uint16_t>,
    RWTextureView<2, Vector2us>,
    RWTextureView<2, Vector3us>,
    RWTextureView<2, Vector4us>,
    // 16-bit Signed
    RWTextureView<2, int16_t>,
    RWTextureView<2, Vector2s>,
    RWTextureView<2, Vector3s>,
    RWTextureView<2, Vector4s>
>;

using SurfRefVariant = Variant
<
    std::monostate,
    // Float
    RWTextureRef<2, Float>,
    RWTextureRef<2, Vector2>,
    RWTextureRef<2, Vector3>,
    RWTextureRef<2, Vector4>,
    // 8-bit Unsigned
    RWTextureRef<2, uint8_t>,
    RWTextureRef<2, Vector2uc>,
    RWTextureRef<2, Vector3uc>,
    RWTextureRef<2, Vector4uc>,
    // 8-bit Signed
    RWTextureRef<2, int8_t>,
    RWTextureRef<2, Vector2c>,
    RWTextureRef<2, Vector3c>,
    RWTextureRef<2, Vector4c>,
    // 16-bit Unsigned
    RWTextureRef<2, uint16_t>,
    RWTextureRef<2, Vector2us>,
    RWTextureRef<2, Vector3us>,
    RWTextureRef<2, Vector4us>,
    // 16-bit Signed
    RWTextureRef<2, int16_t>,
    RWTextureRef<2, Vector2s>,
    RWTextureRef<2, Vector3s>,
    RWTextureRef<2, Vector4s>
>;

// We do this just to forward declare
// TODO: Inclusion of files due to GPU/CPU only code
// is a mess currently. We need to restructure the headers etc.
// hopefully sometime soon.
struct TracerSurfView : public SurfViewVariant
{
    using SurfViewVariant::SurfViewVariant;
};

struct TracerSurfRef : public SurfRefVariant
{
    using SurfRefVariant::SurfRefVariant;
};

// This will create problem though, "VariantSize" could not
// get resolved. We overload them
namespace VariantDetail
{
    template<>
    struct VariantSizeV<TracerSurfView>
        : std::integral_constant<size_t, SurfViewVariant::TypeCount>
    {};

    template<>
    struct VariantSizeV<TracerSurfRef>
        : std::integral_constant<size_t, SurfRefVariant::TypeCount>
    {};
}

static_assert(SurfViewVariant::TD, "\"SurfViewVariant\" is GPU-facing."
              " It's alternatives must be all trivially destructible");