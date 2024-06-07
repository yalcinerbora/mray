#pragma once

//#include <cuda>

#include <cstdint>
#include "Core/TypeFinder.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include <cuda.h>

namespace mray::cuda
{

template <class Query, class Mapped>
struct TexTypeMapping
{
    using KeyType = Query;
    using MappedType = Mapped;
};

template <class Query, uint32_t C>
struct MappingInt
{
    using KeyType = Query;
    static constexpr uint32_t Integer = C;
};

template<class T>
constexpr auto VectorTypeToCUDA()
{
    constexpr auto CUDAToTexTypeMapping = std::make_tuple
    (
        // Unsigned Int
        TexTypeMapping<uint32_t,    uint32_t>{},
        TexTypeMapping<Vector2ui,   uint2>{},
        TexTypeMapping<Vector3ui,   uint4>{},
        TexTypeMapping<Vector4ui,   uint4>{},
        // Int
        TexTypeMapping<int32_t,     int32_t>{},
        TexTypeMapping<Vector2i,    int2>{},
        TexTypeMapping<Vector3i,    int4>{},
        TexTypeMapping<Vector4i,    int4>{},
        // Unsigned Short
        TexTypeMapping<uint16_t,    uint16_t>{},
        TexTypeMapping<Vector2us,   ushort2>{},
        TexTypeMapping<Vector3us,   ushort4>{},
        TexTypeMapping<Vector4us,   ushort4>{},
        // Short
        TexTypeMapping<int16_t,     int16_t>{},
        TexTypeMapping<Vector2s,    short2>{},
        TexTypeMapping<Vector3s,    short4>{},
        TexTypeMapping<Vector4s,    short4>{},
        // Unsigned Char
        TexTypeMapping<uint8_t,     uint8_t>{},
        TexTypeMapping<Vector2uc,   uchar2>{},
        TexTypeMapping<Vector3uc,   uchar4>{},
        TexTypeMapping<Vector4uc,   uchar4>{},
        // Char
        TexTypeMapping<int8_t,      int8_t>{},
        TexTypeMapping<Vector2c,    char2>{},
        TexTypeMapping<Vector3c,    char4>{},
        TexTypeMapping<Vector4c,    char4>{},
        // Float
        TexTypeMapping<float,       float>{},
        TexTypeMapping<Vector2f,    float2>{},
        TexTypeMapping<Vector3f,    float4>{},
        TexTypeMapping<Vector4f,    float4>{}
    );

    using namespace TypeFinder;
    constexpr auto FList = CUDAToTexTypeMapping;
    return GetTupleElement<T>(FList);
}

template<class T>
constexpr auto BCTypeToCUDA()
{
    constexpr auto BCToTexTypeMapping = std::make_tuple
    (
        // Unsigned Int
        MappingInt<PixelBC1, cudaChannelFormatKindUnsignedBlockCompressed1>{},
        MappingInt<PixelBC2, cudaChannelFormatKindUnsignedBlockCompressed2>{},
        MappingInt<PixelBC3, cudaChannelFormatKindUnsignedBlockCompressed3>{},
        MappingInt<PixelBC4U, cudaChannelFormatKindUnsignedBlockCompressed4>{},
        MappingInt<PixelBC4S, cudaChannelFormatKindSignedBlockCompressed4>{},
        MappingInt<PixelBC5U, cudaChannelFormatKindUnsignedBlockCompressed5>{},
        MappingInt<PixelBC5S, cudaChannelFormatKindSignedBlockCompressed5>{},
        MappingInt<PixelBC6U, cudaChannelFormatKindUnsignedBlockCompressed6H>{},
        MappingInt<PixelBC6S, cudaChannelFormatKindSignedBlockCompressed6H>{},
        MappingInt<PixelBC7, cudaChannelFormatKindUnsignedBlockCompressed7>{}
    );
    using namespace TypeFinder;
    constexpr auto FList = BCToTexTypeMapping;
    return GetTupleElement<T>(FList);
}


template<class T>
static constexpr auto VectorTypeToChannels()
{
    constexpr auto CUDAToChannelCountMapping = std::make_tuple
    (
        // 1 Channel
        // Unsigned Int
        MappingInt<uint32_t,    1>{},
        MappingInt<Vector2ui,   2>{},
        MappingInt<Vector3ui,   3>{},
        MappingInt<Vector4ui,   4>{},
        // Int
        MappingInt<int32_t,     1>{},
        MappingInt<Vector2i,    2>{},
        MappingInt<Vector3i,    3>{},
        MappingInt<Vector4i,    4>{},
        // Unsigned Short
        MappingInt<uint16_t,    1>{},
        MappingInt<Vector2us,   2>{},
        MappingInt<Vector3us,   3>{},
        MappingInt<Vector4us,   4>{},
        // Short
        MappingInt<int16_t,     1>{},
        MappingInt<Vector2s,    2>{},
        MappingInt<Vector3s,    3>{},
        MappingInt<Vector4s,    4>{},
        // Unsigned Char
        MappingInt<uint8_t,     1>{},
        MappingInt<Vector2uc,   2>{},
        MappingInt<Vector3uc,   3>{},
        MappingInt<Vector4uc,   4>{},
        // Char
        MappingInt<int8_t,      1>{},
        MappingInt<Vector2c,    2>{},
        MappingInt<Vector3c,    3>{},
        MappingInt<Vector4c,    4>{},
        // Float
        MappingInt<float,       1>{},
        MappingInt<Vector2f,    2>{},
        MappingInt<Vector3f,    3>{},
        MappingInt<Vector4f,    4>{}
    );

    using namespace TypeFinder;
    constexpr auto FList = CUDAToChannelCountMapping;
    return GetTupleElement<T>(FList);
}

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
    static constexpr uint32_t Channels = VectorTypeToChannels<T>().Integer;
    private:
    using CudaType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
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
    static constexpr uint32_t Channels = VectorTypeToChannels<T>().Integer;
    private:
    using CudaType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
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
    static constexpr uint32_t Channels = VectorTypeToChannels<T>().Integer;
    private:
    using CudaType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
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