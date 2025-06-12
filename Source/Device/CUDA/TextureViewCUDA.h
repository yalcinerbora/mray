#pragma once

//#include <cuda>

#include <cstdint>
#include "Core/TypeFinder.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include "Device/GPUTypes.h"
#include <cuda.h>

namespace mray::cuda
{

using VectorTypeToCUDA = TypeFinder::T_TMapper:: template Map
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

template <class MRayType, class CudaType, uint32_t Channels>
struct CudaTexToMRayType {};

template <class MRayType, class CudaType>
struct CudaTexToMRayType<MRayType, CudaType, 1>
{
    static constexpr MRayType Convert(CudaType scalar)
    {
        return scalar;
    }
};

template <class MRayType, class CudaType>
struct CudaTexToMRayType<MRayType, CudaType, 2>
{
    static constexpr MRayType Convert(CudaType vec)
    {
        return MRayType(vec.x, vec.y);
    }
};

template <class MRayType, class CudaType>
struct CudaTexToMRayType<MRayType, CudaType, 3>
{
    static constexpr MRayType Convert(CudaType vec)
    {
        return MRayType(vec.x, vec.y, vec.z);
    }
};

template <class MRayType, class CudaType>
struct CudaTexToMRayType<MRayType, CudaType, 4>
{
    static constexpr MRayType Convert(CudaType vec)
    {
        return MRayType(vec.x, vec.y, vec.z, vec.w);
    }
};

template <class MRayType, class CudaType, uint32_t Channels>
struct MRayToCudaTexType {};

template <class MRayType, class CudaType>
struct MRayToCudaTexType<MRayType, CudaType, 1>
{
    static constexpr CudaType Convert(MRayType scalar)
    {
        return scalar;
    }
};

template <class MRayType, class CudaType>
struct MRayToCudaTexType<MRayType, CudaType, 2>
{
    static constexpr CudaType Convert(MRayType vec)
    {
        return CudaType{.x = vec[0], .y = vec[1]};
    }
};

template <class MRayType, class CudaType>
struct MRayToCudaTexType<MRayType, CudaType, 3>
{
    static constexpr CudaType Convert(MRayType vec)
    {
        return CudaType{.x = vec[0], .y = vec[1], .z = vec[2]};
    }
};

template <class MRayType, class CudaType>
struct MRayToCudaTexType<MRayType, CudaType, 4>
{
    static constexpr CudaType Convert(MRayType vec)
    {
        return CudaType{.x = vec[0], .y = vec[1], .z = vec[2], .w = vec[3]};
    }
};

template<uint32_t DIM, class T>
class TextureViewCUDA
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = DIM;

    private:
    using CudaType      = typename VectorTypeToCUDA::Find<T>;
    using ConvertType   = CudaTexToMRayType<T, CudaType, Channels>;
    using UV            = UVType<DIM>;

    private:
    cudaTextureObject_t     texHandle;

    public:
    MRAY_HOST   TextureViewCUDA(cudaTextureObject_t t) : texHandle(t) {}
    // Base Access
    MRAY_GPU T  operator()(UV uv) const;
    // Gradient Access
    MRAY_GPU T  operator()(UV uv,
                           UV dpdx,
                           UV dpdy) const;
    // Direct Mip Access
    MRAY_GPU T  operator()(UV uv, Float mipLevel) const;
};

// Writable texture views (disregards normalization etc)
template<uint32_t DIM, class T>
class RWTextureViewCUDA
{
    public:
    class PixRef
    {
        friend class RWTextureViewCUDA<DIM, T>;
        public:
        // Texture channels
        static constexpr uint32_t Channels = VectorTypeToChannels<T>();
        // Read channels (3Channel textures are actual 4 channel)
        using PaddedChannelType = PaddedChannel<Channels, T>;
        using PaddedCudaType = typename VectorTypeToCUDA::template Find<PaddedChannelType>;
        using WriteConvertType = MRayToCudaTexType<T, PaddedCudaType, Channels>;

        private:
        cudaSurfaceObject_t surfHandle;
        TextureExtent<DIM>  ij;
        // Constructor
        MRAY_GPU            PixRef(cudaSurfaceObject_t s,
                                   TextureExtent<DIM> ij);
        public:
        MRAY_GPU PixRef&    operator=(const T&);
    };

    static constexpr uint32_t Channels = PixRef::Channels;
    using Type = T;
    using ReadConvertType = CudaTexToMRayType<T, typename PixRef::PaddedCudaType,
                                              Channels>;
    using PaddedChannelType = typename PixRef::PaddedChannelType;

    private:
    cudaSurfaceObject_t surfHandle;

    public:
    // Full Texture object access
    MRAY_HOST       RWTextureViewCUDA(cudaSurfaceObject_t t) : surfHandle(t) {}
    // Write
    MRAY_GPU PixRef operator()(TextureExtent<DIM>);
    // Read
    MRAY_GPU T      operator()(TextureExtent<DIM>) const;
};

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCUDA<D, T>::operator()(UV uv) const
{
    bool isResident = false;
    CudaType t;
    if constexpr(D == 1)
    {
        isResident = true;
        t = tex1D<CudaType>(texHandle, uv);
    }
    else if constexpr(D == 2)
    {
        t = tex2D<CudaType>(texHandle, uv[0], uv[1], &isResident);
    }
    else
    {
        t = tex3D<CudaType>(texHandle,
                            uv[0], uv[1], uv[2],
                            &isResident);
    }
    if(isResident) return ConvertType::Convert(t);
    return T(NAN);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCUDA<D, T>::operator()(UV uv,
                                    UV dpdx,
                                    UV dpdy) const
{
    bool isResident = false;
    CudaType t;
    if constexpr(D == 1)
    {
        isResident = true;
        t = tex1DGrad<CudaType>(texHandle,
                                uv, dpdx, dpdy);
    }
    else if constexpr(D == 2)
    {
        t = tex2DGrad<CudaType>(texHandle, uv[0], uv[1],
                                {dpdx[0], dpdx[1]},
                                {dpdy[0], dpdy[1]}, &isResident);
    }
    else
    {
        t = tex3DGrad<CudaType>(texHandle,
                                uv[0], uv[1], uv[2],
                                {dpdx[0], dpdx[1], dpdx[2]},
                                {dpdy[0], dpdy[1], dpdy[2]},
                                &isResident);
    }
    if(isResident) return ConvertType::Convert(t);
    return T(NAN);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCUDA<D, T>::operator()(UV uv, Float mipLevel) const
{
    bool isResident = false;
    CudaType t;
    if constexpr(D == 1)
    {
        isResident = true;
        t = tex1DLod<CudaType>(texHandle, uv, mipLevel);
    }
    else if constexpr(D == 2)
    {
        t = tex2DLod<CudaType>(texHandle, uv[0], uv[1], mipLevel,
                               &isResident);
    }
    else
    {
        t = tex3DLod<CudaType>(texHandle,
                               uv[0], uv[1], uv[2],
                               mipLevel,
                               &isResident);
    }
    if(isResident) return ConvertType::Convert(t);
    return T(NAN);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
RWTextureViewCUDA<D, T>::PixRef::PixRef(cudaSurfaceObject_t s,
                                        TextureExtent<D> in)
    : surfHandle(s)
    , ij(in)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewCUDA<D, T>::PixRef&
RWTextureViewCUDA<D, T>::PixRef::operator=(const T& val)
{
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");

    // Do some sanity check
    static_assert(sizeof(PaddedChannelType) == sizeof(PaddedCudaType));
    constexpr int size = sizeof(PaddedChannelType);
    // We will read 4 channel data if texture is 3 channel
    PaddedCudaType t;
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
        surf1Dwrite<PaddedCudaType>
        (
            t, surfHandle,
            static_cast<int>(ij) * size
        );
    }
    else if constexpr(D == 2)
    {
        surf2Dwrite<PaddedCudaType>
        (
            t, surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1]
        );
    }
    else if constexpr(D == 3)
    {
        surf3Dwrite<PaddedCudaType>
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
typename RWTextureViewCUDA<D, T>::PixRef
RWTextureViewCUDA<D, T>::operator()(TextureExtent<D> ij)
{
    return PixRef(surfHandle, ij);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T RWTextureViewCUDA<D, T>::operator()(TextureExtent<D> ij) const
{
    using PaddedCudaType = typename PixRef::PaddedCudaType;
    constexpr int size = sizeof(PaddedCudaType);

    PaddedCudaType t;
    if constexpr(D == 1)
    {
        t = surf1DRead<PaddedCudaType>
        (
            surfHandle,
            static_cast<int>(ij) * size
        );
    }
    else if constexpr(D == 2)
    {
        t = surf2Dread<PaddedCudaType>
        (
            surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1]
        );
    }
    else if constexpr(D == 3)
    {
        t = surf3Dread<PaddedCudaType>
        (
            surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1], ij[2]
        );
    }
    else static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
    return T(ReadConvertType::Convert(t));
}

}