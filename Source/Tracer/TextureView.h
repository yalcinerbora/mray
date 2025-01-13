#pragma once

#include "Core/Map.h"
#include "Core/DeviceVisit.h"
#include "Core/TracerI.h"

#include "Device/GPUTextureView.h"

// Wrapper around the texture view of the
// hardware. These views have postprocess
// capability. In order to make this
// statically compile (no dynamic polymorphism),
// everything is in switch/case statements.
// So it is not that extensible.
enum class TextureReadMode : uint8_t
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
    // TODO: We are wasting 8 bytes of memory when this is
    // variant (padding etc. issues)
    // This struct is 16 bytes when we do C unions.
    // Because of this we need to define our constructors
    union
    {
        TextureView<DIM, Float>     tF;
        TextureView<DIM, Vector2>   tV2;
        TextureView<DIM, Vector3>   tV3;
        TextureView<DIM, Vector4>   tV4;
    };
    // TODO: coalesce these reads (compiler does 1 byte reads here)
    // Better to get a word (32-bit) and do bit stuff maybe?
    uint8_t                 index;
    TextureReadMode         mode;
    bool                    flipY = false;

    template<class ReadType>
    MRAY_GPU Optional<T>    Postprocess(Optional<ReadType>&&) const;

    public:
    // Constructors & Desructor
    MRAY_HOST               TracerTexView(HWTextureView<DIM> hwView,
                                          TextureReadMode mode,
                                          bool flipY = false);
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
                Float z = Math::SqrtMax(Float(1) - val[0] * val[0] - val[1] * val[1]);
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
            default: break;
        }
    }
    return std::nullopt;
}

template<uint32_t D, class T>
MRAY_HOST
TracerTexView<D, T>::TracerTexView(HWTextureView<D> hw,
                                   TextureReadMode m,
                                   bool flipYIn)
    : index(static_cast<uint8_t>(hw.index()))
    , mode(m)
    , flipY(flipYIn)
{
    static_assert(std::variant_size_v<HWTextureView<D>> == 4);
    switch(index)
    {
        case 0: tF  = std::get<0>(hw); break;
        case 1: tV2 = std::get<1>(hw); break;
        case 2: tV3 = std::get<2>(hw); break;
        case 3: tV4 = std::get<3>(hw); break;
    }
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::operator()(UV uv) const
{
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: return Postprocess(tF (uv));
        case 1: return Postprocess(tV2(uv));
        case 2: return Postprocess(tV3(uv));
        case 3: return Postprocess(tV4(uv));
    }
    return std::nullopt;
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::operator()(UV uv,
                                            UV dpdx,
                                            UV dpdy) const
{
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: return Postprocess(tF (uv, dpdx, dpdy));
        case 1: return Postprocess(tV2(uv, dpdx, dpdy));
        case 2: return Postprocess(tV3(uv, dpdx, dpdy));
        case 3: return Postprocess(tV4(uv, dpdx, dpdy));
    }
    return std::nullopt;
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TracerTexView<D, T>::operator()(UV uv, Float mipLevel) const
{
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: return Postprocess(tF (uv, mipLevel));
        case 1: return Postprocess(tV2(uv, mipLevel));
        case 2: return Postprocess(tV3(uv, mipLevel));
        case 3: return Postprocess(tV4(uv, mipLevel));
    }
    return std::nullopt;
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