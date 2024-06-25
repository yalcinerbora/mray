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
        return MRayType{.x = vec[0], .y = vec[1]};
    }
};

template <class MRayType, class CudaType>
struct MRayToCudaTexType<MRayType, CudaType, 3>
{
    static constexpr CudaType Convert(MRayType vec)
    {
        return MRayType{.x = vec[0], .y = vec[1], .z = vec[2]};
    }
};

template <class MRayType, class CudaType>
struct MRayToCudaTexType<MRayType, CudaType, 4>
{
    static constexpr CudaType Convert(MRayType vec)
    {
        return MRayType{.x = vec[0], .y = vec[1], .z = vec[2], .w = vec[3]};
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
    MRAY_GPU Optional<T>    operator()(Float uv) const;
    // Gradient Access
    MRAY_GPU Optional<T>    operator()(Float uv, Float dpdx,
                                       Float dpdy) const;
    // Direct Mip Access
    MRAY_GPU Optional<T>    operator()(Float uv, Float mipLevel) const;
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
    MRAY_GPU Optional<T>    operator()(Vector2 uv) const;
    // Gradient Access
    MRAY_GPU Optional<T>    operator()(Vector2 uv, Vector2 dpdx,
                                       Vector2 dpdy) const;
    // Direct Mip Access
    MRAY_GPU Optional<T>    operator()(Vector2 uv, Float mipLevel) const;
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
    MRAY_GPU Optional<T>    operator()(Vector3 uv) const;
    // Gradient Access
    MRAY_GPU Optional<T>    operator()(Vector3 uv, Vector3 dpdx,
                                       Vector3 dpdy) const;
    // Direct Mip Access
    MRAY_GPU Optional<T>    operator()(Vector3 uv, Float mipLevel) const;
};

// Writable texture views (disregards normalization etc)
template<uint32_t DIM, class T>
class RWTextureViewCUDA
{
    public:
    class PixRef
    {
        // Texture channels
        static constexpr uint32_t Channels = VectorTypeToChannels::Find<T>;
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

    private:
    using ReadConvertType = CudaTexToMRayType<T, typename PixRef::PaddedCudaType,
                                              PixRef::Channels>;

    private:
    cudaSurfaceObject_t     surfHandle;

    public:
    // Full Texture object access
    MRAY_HOST               RWTextureViewCUDA(cudaSurfaceObject_t t) : surfHandle(t) {}
    // Write
    MRAY_GPU PixRef         operator()(TextureExtent<DIM>);
    // Read
    MRAY_GPU Optional<T>    operator()(TextureExtent<DIM>) const;
};

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv) const
{
    CudaType t = tex1D<CudaType>(texHandle, uv);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv, Float dpdx,
                                              Float dpdy) const
{
    CudaType t = tex1DGrad<CudaType>(texHandle,
                                     uv, dpdx, dpdy);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv, Float mipLevel) const
{
    CudaType t = tex1DLod<CudaType>(texHandle, uv, mipLevel);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv) const
{
    bool isResident = false;
    CudaType t = tex2D<CudaType>(texHandle, uv[0], uv[1], &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv, Float mipLevel) const
{
    bool isResident = false;
    CudaType t = tex2DLod<CudaType>(texHandle, uv[0], uv[1], mipLevel,
                                    &isResident);
    if(isResident) return ConvertType::Convert(t);
    return std::nullopt;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
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
        surf1DWrite<PaddedCudaType>
        (
            t, surfHandle,
            static_cast<int>(ij) * size
        );
    }
    else if constexpr(D == 2)
    {
        surf2DWrite<PaddedCudaType>
        (
            t, surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1]
        );
    }
    else if constexpr(D == 3)
    {
        surf3DWrite<PaddedCudaType>
        (
            t, surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1], ij[2]
        );
    }
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewCUDA<D, T>::PixRef
RWTextureViewCUDA<D, T>::operator()(TextureExtent<D> ij)
{
    return PixRef(*this, ij);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T> RWTextureViewCUDA<D, T>::operator()(TextureExtent<D> ij) const
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
        t = surf2DWrite<PaddedCudaType>
        (
            surfHandle,
            static_cast<int>(ij[0]) * size,
            ij[1]
        );
    }
    else if constexpr(D == 3)
    {
        t = surf3DWrite<PaddedCudaType>
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