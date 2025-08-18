#pragma once
#include "Core/TracerI.h"
#include "Core/Map.h"
#include "Core/Variant.h"

#include "Device/GPUTextureView.h"

#include "StreamingTextureView.h"

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
    // Given 2-channel **signed** format,
    // calculate Z channel,
    // abuse |N| = 1. (Usefull for BC5s)
    TO_3C_BASIC_NORMAL_FROM_SIGNED_2C,
    // Given 2-channel **signed** format,
    // calculate Z channel,
    // assume x/y = [0,1] and
    // do concentric octahedral mapping
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
class TracerTexView;

template<class T>
class TracerTexView<2, T>
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = 2;
    private:
    using UV = UVType<2>;

    private:
    // TODO: We are wasting 8 bytes of memory when this is
    // variant (padding etc. issues)
    // This struct is 16 bytes when we do C unions.
    // Because of this we need to define our constructors
    union
    {
        TextureView<2, Float>   tF;
        TextureView<2, Vector2> tV2;
        TextureView<2, Vector3> tV3;
        TextureView<2, Vector4> tV4;
        StreamingTextureView    sT;
    };
    // TODO: coalesce these reads (compiler does 1 byte reads here)
    // Better to get a word (32-bit) and do bit stuff maybe?
    uint8_t         index;
    TextureReadMode mode;
    bool            flipY = false;

    public:
    // Constructors & Destructor
    MRAY_HOST   TracerTexView(HWTextureView<2> hwView,
                              TextureReadMode mode,
                                bool flipY = false);
    MRAY_HOST   TracerTexView(StreamingTextureView sTView,
                              TextureReadMode mode,
                              bool flipY = false);

    // Base Access
    MR_GF_DECL T    operator()(UV uv) const;
    // Gradient Access
    MR_GF_DECL T    operator()(UV uv,
                               UV dpdx,
                               UV dpdy) const;

    // Direct Mip Access
    MR_GF_DECL T    operator()(UV uv, Float mipLevel) const;

    // Texture Residency check
    MR_GF_DECL bool IsResident(UV uv, Float mipLevel) const;
    MR_GF_DECL bool IsResident(UV uv, UV dpdx, UV dpdy) const;
    MR_GF_DECL bool IsResident(UV uv) const;
};

template<class T>
class TracerTexView<3, T>
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = 3;
    private:
    using UV = UVType<3>;

    private:
    // TODO: We are wasting 8 bytes of memory when this is
    // variant (padding etc. issues)
    // This struct is 16 bytes when we do C unions.
    // Because of this we need to define our constructors
    union
    {
        TextureView<3, Float>     tF;
        TextureView<3, Vector2>   tV2;
        TextureView<3, Vector3>   tV3;
        TextureView<3, Vector4>   tV4;
    };
    // TODO: coalesce these reads (compiler does 1 byte reads here)
    // Better to get a word (32-bit) and do bit stuff maybe?
    uint8_t             index;
    TextureReadMode     mode;
    bool                flipY = false;

    public:
    // Constructors & Destructor
    MRAY_HOST   TracerTexView(HWTextureView<3> hwView,
                              TextureReadMode mode,
                              bool flipY = false);
    // Base Access
    MR_GF_DECL T    operator()(UV uv) const;
    // Gradient Access
    MR_GF_DECL T    operator()(UV uv,
                               UV dpdx,
                               UV dpdy) const;
    // Direct Mip Access
    MR_GF_DECL T    operator()(UV uv, Float mipLevel) const;

    // Texture Residency check
    MR_GF_DECL bool IsResident(UV, Float) const { return true; }
    MR_GF_DECL bool IsResident(UV, UV, UV) const { return true; }
    MR_GF_DECL bool IsResident(UV) const { return true; }
};

template<class T>
MRAY_HOST
TracerTexView<2, T>::TracerTexView(HWTextureView<2> hw,
                                   TextureReadMode m,
                                   bool flipYIn)
    : index(static_cast<uint8_t>(hw.index()))
    , mode(m)
    , flipY(flipYIn)
{
    static_assert(std::variant_size_v<HWTextureView<2>> == 4);
    switch(index)
    {
        case 0: tF  = std::get<0>(hw); break;
        case 1: tV2 = std::get<1>(hw); break;
        case 2: tV3 = std::get<2>(hw); break;
        case 3: tV4 = std::get<3>(hw); break;
    }
}

template<class T>
MRAY_HOST
TracerTexView<2, T>::TracerTexView(StreamingTextureView sTView,
                                   TextureReadMode m,
                                   bool flipYIn)
    : index(uint8_t{4})
    , mode(m)
    , flipY(flipYIn)
    , sT(sTView)
{}

template<class T>
MRAY_HOST
TracerTexView<3, T>::TracerTexView(HWTextureView<3> hw,
                                   TextureReadMode m,
                                   bool flipYIn)
    : index(static_cast<uint8_t>(hw.index()))
    , mode(m)
    , flipY(flipYIn)
{
    static_assert(std::variant_size_v<HWTextureView<3>> == 4);
    switch(index)
    {
        case 0: tF  = std::get<0>(hw); break;
        case 1: tV2 = std::get<1>(hw); break;
        case 2: tV3 = std::get<2>(hw); break;
        case 3: tV4 = std::get<3>(hw); break;
    }
}

// Texture Related types
using GenericTextureView = Variant//std::variant
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