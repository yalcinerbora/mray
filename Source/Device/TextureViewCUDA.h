#pragma once

//#include <cuda>

#include <cstdint>
#include "Core/TypeFinder.h"
#include "Core/Vector.h"
#include "Core/Types.h"
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


using VectorTypeToChannels = TypeFinder::T_VMapper:: template Map
<
    // 1 Channel
    // Unsigned Int
    TypeFinder::T_VMapper::template TVPair<uint32_t,    1>,
    TypeFinder::T_VMapper::template TVPair<Vector2ui,   2>,
    TypeFinder::T_VMapper::template TVPair<Vector3ui,   3>,
    TypeFinder::T_VMapper::template TVPair<Vector4ui,   4>,
    // Int
    TypeFinder::T_VMapper::template TVPair<int32_t,     1>,
    TypeFinder::T_VMapper::template TVPair<Vector2i,    2>,
    TypeFinder::T_VMapper::template TVPair<Vector3i,    3>,
    TypeFinder::T_VMapper::template TVPair<Vector4i,    4>,
    // Unsigned Short
    TypeFinder::T_VMapper::template TVPair<uint16_t,    1>,
    TypeFinder::T_VMapper::template TVPair<Vector2us,   2>,
    TypeFinder::T_VMapper::template TVPair<Vector3us,   3>,
    TypeFinder::T_VMapper::template TVPair<Vector4us,   4>,
    // Short
    TypeFinder::T_VMapper::template TVPair<int16_t,     1>,
    TypeFinder::T_VMapper::template TVPair<Vector2s,    2>,
    TypeFinder::T_VMapper::template TVPair<Vector3s,    3>,
    TypeFinder::T_VMapper::template TVPair<Vector4s,    4>,
    // Unsigned Char
    TypeFinder::T_VMapper::template TVPair<uint8_t,     1>,
    TypeFinder::T_VMapper::template TVPair<Vector2uc,   2>,
    TypeFinder::T_VMapper::template TVPair<Vector3uc,   3>,
    TypeFinder::T_VMapper::template TVPair<Vector4uc,   4>,
    // Char
    TypeFinder::T_VMapper::template TVPair<int8_t,      1>,
    TypeFinder::T_VMapper::template TVPair<Vector2c,    2>,
    TypeFinder::T_VMapper::template TVPair<Vector3c,    3>,
    TypeFinder::T_VMapper::template TVPair<Vector4c,    4>,
    // Float
    TypeFinder::T_VMapper::template TVPair<float,       1>,
    TypeFinder::T_VMapper::template TVPair<Vector2f,    2>,
    TypeFinder::T_VMapper::template TVPair<Vector3f,    3>,
    TypeFinder::T_VMapper::template TVPair<Vector4f,    4>
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

template<uint32_t DIM, class T>
class TextureViewCUDA;

// Specializiation of 2D for CUDA
template<class T>
class TextureViewCUDA<1, T>
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels::Find<T>;
    private:
    using CudaType = typename VectorTypeToCUDA::Find<T>;
    using ConvertType = CudaTexToMRayType<T, CudaType, Channels>;

    private:
    cudaTextureObject_t     texHandle;

    public:
    // Full Texture object access
    MRAY_HOST               TextureViewCUDA(cudaTextureObject_t t) : texHandle(t) {}


    // Base Access
    MRAY_HYBRID Optional<T> operator()(Float uv) const;
    // Gradient Access
    MRAY_HYBRID Optional<T> operator()(Float uv, Float dpdx,
                                       Float dpdy) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<T> operator()(Float uv, Float mipLevel) const;
};

// Specializiations of 123D for CUDA
template<class T>
class TextureViewCUDA<2, T>
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels::Find<T>;
    private:
    using CudaType = typename VectorTypeToCUDA::Find<T>;
    using ConvertType = CudaTexToMRayType<T, CudaType, Channels>;

    private:
    cudaTextureObject_t     texHandle;

    public:
    MRAY_HOST               TextureViewCUDA(cudaTextureObject_t t) : texHandle(t) {}
    // Base Access
    MRAY_HYBRID Optional<T> operator()(Vector2 uv) const;
    // Gradient Access
    MRAY_HYBRID Optional<T> operator()(Vector2 uv, Vector2 dpdx,
                                       Vector2 dpdy) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<T> operator()(Vector2 uv, Float mipLevel) const;
};

template<class T>
class TextureViewCUDA<3, T>
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels::Find<T>;
    private:
    using CudaType = typename VectorTypeToCUDA::Find<T>;
    using ConvertType = CudaTexToMRayType<T, CudaType, Channels>;

    private:
    cudaTextureObject_t     texHandle;

    public:
    MRAY_HOST               TextureViewCUDA(cudaTextureObject_t t) : texHandle(t) {}
    // Base Access
    MRAY_HYBRID Optional<T> operator()(Vector3 uv) const;
    // Gradient Access
    MRAY_HYBRID Optional<T> operator()(Vector3 uv, Vector3 dpdx,
                                       Vector3 dpdy) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<T> operator()(Vector3 uv, Float mipLevel) const;
};

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv) const
{
    CudaType t = tex1D<CudaType>(texHandle, uv);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv, Float dpdx,
                                              Float dpdy) const
{
    CudaType t = tex1DGrad<CudaType>(texHandle,
                                     uv, dpdx, dpdy);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv, Float mipLevel) const
{
    CudaType t = tex1DLod<CudaType>(texHandle, uv, mipLevel);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv) const
{
    bool isResident = false;
    CudaType t = tex2D<CudaType>(texHandle, uv[0], uv[1], &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv, Vector2 dpdx,
                                              Vector2 dpdy) const
{
    bool isResident = false;
    CudaType t = tex2DGrad<CudaType>(texHandle, uv[0], uv[1],
                                     {dpdx[0], dpdx[1]},
                                     {dpdy[0], dpdy[1]}, &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv, Float mipLevel) const
{
    bool isResident = false;
    CudaType t = tex2DLod<CudaType>(texHandle, uv[0], uv[1], mipLevel,
                                    &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<3, T>::operator()(Vector3 uv) const
{
    bool isResident = false;
    CudaType t = tex3D<CudaType>(texHandle,
                                 uv[0], uv[1], uv[2],
                                 &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<3, T>::operator()(Vector3 uv, Vector3 dpdx,
                                              Vector3 dpdy) const
{
    bool isResident = false;
    CudaType t = tex3DGrad<CudaType>(texHandle,
                                     uv[0], uv[1], uv[2],
                                     {dpdx[0], dpdx[1], dpdx[2]},
                                     {dpdy[0], dpdy[1], dpdy[2]},
                                     &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<3, T>::operator()(Vector3 uv, Float mipLevel) const
{
    bool isResident = false;
    CudaType t = tex3DLod<CudaType>(texHandle,
                                    uv[0], uv[1], uv[2],
                                    mipLevel,
                                    &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

}