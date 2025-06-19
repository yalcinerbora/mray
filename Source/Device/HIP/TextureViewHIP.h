#pragma once

#include <cstdint>
#include "Core/TypeFinder.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include "Device/GPUTypes.h"
#include <hip/hip_runtime.h>

namespace mray::hip
{

using VectorTypeToHIP = TypeFinder::T_TMapper:: template Map
<
    // Unsigned Int
    TypeFinder::T_TMapper::template TTPair<uint32_t,    uint32_t>,
    TypeFinder::T_TMapper::template TTPair<Vector2ui,   uint2>,
    TypeFinder::T_TMapper::template TTPair<Vector3ui,   uint4>,
    TypeFinder::T_TMapper::template TTPair<Vector4ui,   uint4>,
    // Int
    TypeFinder::T_TMapper::template TTPair<int32_t,     int32_t>,
    TypeFinder::T_TMapper::template TTPair<Vector2i,    int2>,
    TypeFinder::T_TMapper::template TTPair<Vector3i,    int4>,
    TypeFinder::T_TMapper::template TTPair<Vector4i,    int4>,
    // Unsigned Short
    TypeFinder::T_TMapper::template TTPair<uint16_t,    uint16_t>,
    TypeFinder::T_TMapper::template TTPair<Vector2us,   ushort2>,
    TypeFinder::T_TMapper::template TTPair<Vector3us,   ushort4>,
    TypeFinder::T_TMapper::template TTPair<Vector4us,   ushort4>,
    // Short
    TypeFinder::T_TMapper::template TTPair<int16_t,     int16_t>,
    TypeFinder::T_TMapper::template TTPair<Vector2s,    short2>,
    TypeFinder::T_TMapper::template TTPair<Vector3s,    short4>,
    TypeFinder::T_TMapper::template TTPair<Vector4s,    short4>,
    // Unsigned Char
    TypeFinder::T_TMapper::template TTPair<uint8_t,     uint8_t>,
    TypeFinder::T_TMapper::template TTPair<Vector2uc,   uchar2>,
    TypeFinder::T_TMapper::template TTPair<Vector3uc,   uchar4>,
    TypeFinder::T_TMapper::template TTPair<Vector4uc,   uchar4>,
    // Char
    TypeFinder::T_TMapper::template TTPair<int8_t,      int8_t>,
    TypeFinder::T_TMapper::template TTPair<Vector2c,    char2>,
    TypeFinder::T_TMapper::template TTPair<Vector3c,    char4>,
    TypeFinder::T_TMapper::template TTPair<Vector4c,    char4>,
    // Float
    TypeFinder::T_TMapper::template TTPair<float,       float>,
    TypeFinder::T_TMapper::template TTPair<Vector2f,    float2>,
    TypeFinder::T_TMapper::template TTPair<Vector3f,    float4>,
    TypeFinder::T_TMapper::template TTPair<Vector4f,    float4>
>;

template <class MRayType, class HipType, uint32_t Channels>
struct HipTexToMRayType {};

template <class MRayType, class HipType>
struct HipTexToMRayType<MRayType, HipType, 1>
{
    static constexpr MRayType Convert(HipType scalar)
    {
        return scalar;
    }
};

template <class MRayType, class HipType>
struct HipTexToMRayType<MRayType, HipType, 2>
{
    static constexpr MRayType Convert(HipType vec)
    {
        return MRayType(vec.x, vec.y);
    }
};

template <class MRayType, class HipType>
struct HipTexToMRayType<MRayType, HipType, 3>
{
    static constexpr MRayType Convert(HipType vec)
    {
        return MRayType(vec.x, vec.y, vec.z);
    }
};

template <class MRayType, class HipType>
struct HipTexToMRayType<MRayType, HipType, 4>
{
    static constexpr MRayType Convert(HipType vec)
    {
        return MRayType(vec.x, vec.y, vec.z, vec.w);
    }
};

template <class MRayType, class HipType, uint32_t Channels>
struct MRayToHipTexType {};

template <class MRayType, class HipType>
struct MRayToHipTexType<MRayType, HipType, 1>
{
    static constexpr HipType Convert(MRayType scalar)
    {
        return scalar;
    }
};

template <class MRayType, class HipType>
struct MRayToHipTexType<MRayType, HipType, 2>
{
    static constexpr HipType Convert(MRayType vec)
    {
        return HipType(vec[0], vec[1]);
    }
};

template <class MRayType, class HipType>
struct MRayToHipTexType<MRayType, HipType, 3>
{
    static constexpr HipType Convert(MRayType vec)
    {
        return HipType{.x = vec[0], .y = vec[1], .z = vec[2]};
    }
};

template <class MRayType, class HipType>
struct MRayToHipTexType<MRayType, HipType, 4>
{
    static constexpr HipType Convert(MRayType vec)
    {
        return HipType(vec[0], vec[1], vec[2], vec[3]);
    }
};

template<uint32_t DIM, class T>
class TextureViewHIP
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = DIM;

    private:
    using HipType       = typename VectorTypeToHIP::Find<T>;
    using ConvertType   = HipTexToMRayType<T, HipType, Channels>;
    using UV            = UVType<DIM>;

    private:
    hipTextureObject_t      texHandle;

    public:
    MRAY_HOST               TextureViewHIP(hipTextureObject_t t) : texHandle(t) {}
    // Base Access
    MRAY_GPU T              operator()(UV uv) const;
    // Gradient Access
    MRAY_GPU T              operator()(UV uv,
                                       UV dpdx,
                                       UV dpdy) const;
    // Direct Mip Access
    MRAY_GPU T              operator()(UV uv, Float mipLevel) const;
};

// Writable texture views (disregards normalization etc)
template<uint32_t DIM, class T>
class RWTextureViewHIP
{
    public:
    class PixRef
    {
        friend class RWTextureViewHIP<DIM, T>;
        public:
        // Texture channels
        static constexpr uint32_t Channels = VectorTypeToChannels<T>();
        // Read channels (3Channel textures are actual 4 channel)
        using PaddedChannelType = PaddedChannel<Channels, T>;
        using PaddedHipType = typename VectorTypeToHIP::template Find<PaddedChannelType>;
        using WriteConvertType = MRayToHipTexType<T, PaddedHipType, Channels>;

        private:
        hipSurfaceObject_t surfHandle;
        TextureExtent<DIM>  ij;
        // Constructor
        MRAY_GPU            PixRef(hipSurfaceObject_t s,
                                   TextureExtent<DIM> ij);
        public:
        MRAY_GPU PixRef&    operator=(const T&);
    };

    static constexpr uint32_t Channels = PixRef::Channels;
    using Type = T;
    using ReadConvertType = HipTexToMRayType<T, typename PixRef::PaddedHipType,
                                              Channels>;
    using PaddedChannelType = typename PixRef::PaddedChannelType;

    private:
    hipSurfaceObject_t surfHandle;

    public:
    // Full Texture object access
    MRAY_HOST       RWTextureViewHIP(hipSurfaceObject_t t) : surfHandle(t) {}
    // Write
    MRAY_GPU PixRef operator()(TextureExtent<DIM>);
    // Read
    MRAY_GPU T      operator()(TextureExtent<DIM>) const;
};

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewHIP<D, T>::operator()(UV uv) const
{
    HipType t;
    if constexpr(D == 1)
    {
        t = tex1D<HipType>(texHandle, uv);
    }
    else if constexpr(D == 2)
    {
        t = tex2D<HipType>(texHandle, uv[0], uv[1]);
    }
    else
    {
        t = tex3D<HipType>(texHandle,
                            uv[0], uv[1], uv[2]);
    }
    return ConvertType::Convert(t);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewHIP<D, T>::operator()(UV uv, UV dpdx, UV dpdy) const
{
    HipType t;
    if constexpr(D == 1)
    {
        t = tex1DGrad<HipType>(texHandle,
                               uv, dpdx, dpdy);
    }
    else if constexpr(D == 2)
    {
        t = tex2DGrad<HipType>(texHandle, uv[0], uv[1],
                               {dpdx[0], dpdx[1]},
                               {dpdy[0], dpdy[1]});
    }
    else
    {
        t = tex3DGrad<HipType>(texHandle,
                               uv[0], uv[1], uv[2],
                               {dpdx[0], dpdx[1], dpdx[2]},
                               {dpdy[0], dpdy[1], dpdy[2]});
    }
    return ConvertType::Convert(t);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewHIP<D, T>::operator()(UV uv, Float mipLevel) const
{
    HipType t;
    if constexpr(D == 1)
    {
        t = tex1DLod<HipType>(texHandle, uv, mipLevel);
    }
    else if constexpr(D == 2)
    {
        t = tex2DLod<HipType>(texHandle, uv[0], uv[1], mipLevel);
    }
    else
    {
        t = tex3DLod<HipType>(texHandle,
                              uv[0], uv[1], uv[2],
                              mipLevel);
    }
    return ConvertType::Convert(t);    
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
RWTextureViewHIP<D, T>::PixRef::PixRef(hipSurfaceObject_t s,
                                       TextureExtent<D> in)
    : surfHandle(s)
    , ij(in)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewHIP<D, T>::PixRef&
RWTextureViewHIP<D, T>::PixRef::operator=(const T& val)
{
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");

    // Do some sanity check
    static_assert(sizeof(PaddedChannelType) == sizeof(PaddedHipType));
    constexpr int size = sizeof(PaddedChannelType);
    // We will read 4 channel data if texture is 3 channel
    PaddedHipType t;
    // Easy path just do an assignment
    if constexpr(std::is_same_v<PaddedChannelType, T>)
        t = WriteConvertType::Convert(val);
    else
    {
        // Somewhat hard part. All padded types are 3 channel types,
        // so manually assign it to the type T (float3, ushort3 etc..)
        // Again some sanity
        static_assert(VectorC<T> && T::Dims == 3);
        t = {val[0], val[1], val[2], typename T::InnerType(0)};
    }

    if constexpr(D == 1)
    {
        surf1Dwrite<PaddedHipType>
        (
            t, surfHandle,
            static_cast<int>(ij) * size
        );
    }
    else if constexpr(D == 2)
    {
        surf2Dwrite<PaddedHipType>
        (
            t, surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1]
        );
    }
    else if constexpr(D == 3)
    {
        surf3Dwrite<PaddedHipType>
        (
            t, surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1], ij[2]
        );
    }
    return *this;
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewHIP<D, T>::PixRef
RWTextureViewHIP<D, T>::operator()(TextureExtent<D> ij)
{
    return PixRef(surfHandle, ij);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T RWTextureViewHIP<D, T>::operator()(TextureExtent<D> ij) const
{
    using PaddedHipType = typename PixRef::PaddedHipType;
    constexpr int size = sizeof(PaddedHipType);

    PaddedHipType t;
    if constexpr(D == 1)
    {
        surf1DRead<PaddedHipType>
        (
            &t,
            surfHandle,
            static_cast<int>(ij) * size
        );
    }
    else if constexpr(D == 2)
    {
        surf2Dread<PaddedHipType>
        (
            &t,
            surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1]
        );
    }
    else if constexpr(D == 3)
    {
        surf3Dread<PaddedHipType>
        (
            &t,
            surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1], ij[2]
        );
    }
    else static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
    return T(ReadConvertType::Convert(t));
}

}