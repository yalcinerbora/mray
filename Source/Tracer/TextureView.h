#pragma once

#include "Device/GPUSystem.h"
#include "Core/Map.h"
#include "Core/DeviceVisit.h"
#include "Core/TracerI.h"

// Wrapper around the texture view of the
// hardware. These views have postprocess
// capability. In order to make this
// statically compile (no dynamic polymorphism),
// everything is in switch/case statements.
// So it is not that extensible.
enum class TextureReadMode
{
    // Directly read whatever the type is
    DIRECT,
    // Given 2-channel **Signed** format,
    // calculate Z channel,
    // abuse |N| = 1. (Usefull for BC5s)
    TO_3C_BASIC_NORMAL_FROM_SIGNED_2C,
    // Given 2-channel **signed** format,
    // calculate Z channel,
    // assume x/y = [0,1] and
    // do cocentric octrahedral mapping
    // to find xyz
    TO_3C_OCTA_NORMAL_FROM_SIGNED_2C,
    // Similar to the above but before the conversion
    // (x * 2 - 1) is applied to each channel to convert
    // [0, 1] to [-1, 1].
    TO_3C_BASIC_NORMAL_FROM_UNSIGNED_2C,
    TO_3C_OCTA_NORMAL_FROM_UNSIGNED_2C,
    // Channel dropping (only most significand
    // channels are dropped, no swizzling)
    TO_3C_FROM_4C,
    TO_2C_FROM_4C,
    TO_1C_FROM_4C,
    TO_2C_FROM_3C,
    TO_1C_FROM_3C,
    TO_1C_FROM_2C
};

template<uint32_t DIM>
using HWTextureView = Variant
<
    TextureView<DIM, Float>,
    TextureView<DIM, Vector2>,
    TextureView<DIM, Vector3>,
    TextureView<DIM, Vector4>
>;

template<uint32_t DIM, class T>
class TracerTexView
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = DIM;
    private:
    using UV = UVType<DIM>;

    private:
    HWTextureView<DIM>      hwView;
    TextureReadMode         mode;

    template<class ReadType>
    MRAY_GPU Optional<T>    Postprocess(Optional<ReadType>&&) const;

    public:
    // Constructors & Desructor
    MRAY_HOST               TracerTexView(HWTextureView<DIM> hwView,
                                          TextureReadMode mode);
    // Base Access
    MRAY_GPU Optional<T>    operator()(UV uv) const;
    // Gradient Access
    MRAY_GPU Optional<T>    operator()(UV uv,
                                       UV dpdx,
                                       UV dpdy) const;
    // Direct Mip Access
    MRAY_GPU Optional<T>    operator()(UV uv, Float mipLevel) const;
};


template<uint32_t D, class T>
template<class ReadType>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::Postprocess(Optional<ReadType>&& t) const
{
    // Empty optional
    if(!t) return std::nullopt;

    // We specifically do not care about "mode"
    // swicth to reduce the code bloat
    // CPU side will validate the mode and type correctness
    //
    // YOLO Switch/Case + If/Else
    //
    // Same channel
    if constexpr(std::is_same_v<ReadType, T>)
        return t;
    // Channel drop for V1
    if constexpr(std::is_same_v<T, Float> &&
                 (std::is_same_v<ReadType, Vector2> ||
                  std::is_same_v<ReadType, Vector3> ||
                  std::is_same_v<ReadType, Vector4>))
    {
        return Float((*t)[0]);
    }
    // Channel drop for V2
    else if constexpr(std::is_same_v<T, Vector2> &&
                      (std::is_same_v<ReadType, Vector3> ||
                       std::is_same_v<ReadType, Vector4>))
    {
        return Vector2(*t);
    }
    // Channel drop for V3
    else if constexpr(std::is_same_v<T, Vector3> &&
                      std::is_same_v<ReadType, Vector4>)
    {
        return Vector3(*t);
    }
    // Normal mapping conversions
    else if constexpr(std::is_same_v<T, Vector3> &&
                      std::is_same_v<ReadType, Vector2>)
    {
        Vector2 val = *t;
        switch(mode)
        {
            using enum TextureReadMode;
            case TO_3C_BASIC_NORMAL_FROM_UNSIGNED_2C:
            {
                val = val * Vector2(2) - Vector2(1);
            }
            [[fallthrough]];
            case TO_3C_BASIC_NORMAL_FROM_SIGNED_2C:
            {
                Float z = MathFunctions::SqrtMax(Float(1) - val[0] * val[0] - val[1] * val[1]);
                return Vector3(val[0], val[1], z);
            }
            // Octahedral mapping
            // From https://jcgt.org/published/0003/02/01/paper.pdf
            case TO_3C_OCTA_NORMAL_FROM_UNSIGNED_2C:
            {
                val = val * Vector2(2) - Vector2(1);
            }
            [[fallthrough]];
            case TO_3C_OCTA_NORMAL_FROM_SIGNED_2C:
            {
                val = Vector2(val[0] + val[1], val[0] - val[1]) * Float(0.5);
                return Vector3(val, Float(1) - val.Abs().Sum());
            }
        }
    }
    return std::nullopt;
}

template<uint32_t D, class T>
MRAY_HOST
TracerTexView<D, T>::TracerTexView(HWTextureView<D> hw,
                                   TextureReadMode m)
    : hwView(hw)
    , mode(m)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::operator()(UV uv) const
{
    return DeviceVisit(hwView, [&](auto&& view) -> Optional<T>
    {
        return Postprocess(view(uv));
    });
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::operator()(UV uv,
                                            UV dpdx,
                                            UV dpdy) const
{
    return DeviceVisit(hwView, [&](auto&& view) -> Optional<T>
    {
        return Postprocess(view(uv, dpdx, dpdy));
    });
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::operator()(UV uv, Float mipLevel) const
{
    return DeviceVisit(hwView, [&](auto&& view) -> Optional<T>
    {
        return Postprocess(view(uv, mipLevel));
    });
}

// Texture Related types
using GenericTextureView = Variant
<
    TracerTexView<2, Float>,
    TracerTexView<2, Vector2>,
    TracerTexView<2, Vector3>,
    TracerTexView<2, Vector4>,

    TracerTexView<3, Float>,
    TracerTexView<3, Vector2>,
    TracerTexView<3, Vector3>,
    TracerTexView<3, Vector4>
>;
using TextureViewMap = Map<TextureId, GenericTextureView>;