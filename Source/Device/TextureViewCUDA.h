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
struct Mapping
{
    using QueryType = Query;
    using MappedType = Mapped;
};

template <class Query, uint32_t C>
struct MappingInt
{
    using QueryType = Query;
    static constexpr uint32_t Channels = C;
};

template<class T>
static constexpr auto VectorTypeToCUDA()
{
    constexpr auto CUDAToTexTypeMapping = std::make_tuple
    (
        // Unsigned Int
        Mapping<uint32_t,   uint32_t>{},
        Mapping<Vector2ui,  uint2>{},
        Mapping<Vector3ui,  uint4>{},
        Mapping<Vector4ui,  uint4>{},
        // Int
        Mapping<int32_t,    int32_t>{},
        Mapping<Vector2i,   int2>{},
        Mapping<Vector3i,   int4>{},
        Mapping<Vector4i,   int4>{},
        // Unsigned Short
        Mapping<uint16_t,   uint16_t>{},
        Mapping<Vector2us,  ushort2>{},
        Mapping<Vector3us,  ushort4>{},
        Mapping<Vector4us,  ushort4>{},
        // Short
        Mapping<int16_t,    int16_t>{},
        Mapping<Vector2s,   short2>{},
        Mapping<Vector3s,   short4>{},
        Mapping<Vector4s,   short4>{},
        // Unsigned Char
        Mapping<uint8_t,    uint8_t>{},
        Mapping<Vector2uc,  uchar2>{},
        Mapping<Vector3uc,  uchar4>{},
        Mapping<Vector4uc,  uchar4>{},
        // Char
        Mapping<int8_t,     int16_t>{},
        Mapping<Vector2c,   char2>{},
        Mapping<Vector3c,   char4>{},
        Mapping<Vector4c,   char4>{},
        // Float
        Mapping<float,      float>{},
        Mapping<Vector2f,   float2>{},
        Mapping<Vector3f,   float4>{},
        Mapping<Vector4f,   float4>{}
    );

    using namespace TypeFinder;
    constexpr auto FList = CUDAToTexTypeMapping;
    return GetTupleElement<T>(std::forward<decltype(FList)>(FList));
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
    return GetTupleElement<T>(std::forward<decltype(FList)>(FList));
}

template <class MRayType, class CudaType, int Channels>
class CudaTexToMRayType {};

template <class MRayType, class CudaType>
class CudaTexToMRayType<MRayType, CudaType, 1>
{
    static constexpr MRayType Convert(CudaType scalar)
    {
        return scalar;
    }
};

template <class MRayType, class CudaType>
class CudaTexToMRayType<MRayType, CudaType, 2>
{
    static constexpr MRayType Convert(CudaType vec)
    {
        return MRayType(vec.x, vec.y);
    }
};

template <class MRayType, class CudaType>
class CudaTexToMRayType<MRayType, CudaType, 3>
{
    static constexpr MRayType Convert(CudaType vec)
    {
        return MRayType(vec.x, vec.y, vec.z);
    }
};

template <class MRayType, class CudaType>
class CudaTexToMRayType<MRayType, CudaType, 4>
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
    private:
    using MappedType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
    using ConvertType = CudaTexToMRayType<T, MappedType, VectorTypeToChannels<T>()>;
    cudaTextureObject_t texHandle;

    public:
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
    private:
    using MappedType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
    using ConvertType = CudaTexToMRayType<T, MappedType, VectorTypeToChannels<T>()>;
    cudaTextureObject_t texHandle;

    public:
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
    private:
    using MappedType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
    using ConvertType = CudaTexToMRayType<T, MappedType, VectorTypeToChannels<T>()>;
    cudaTextureObject_t texHandle;

    public:
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
    MappedType t = tex2D<MappedType>(texHandle, uv);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv, Float dpdx,
                                              Float dpdy) const
{
    MappedType t = tex1DGrad<MappedType>(texHandle,
                                         uv, dpdx, dpdy);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<1, T>::operator()(Float uv, Float mipLevel) const
{
    MappedType t = tex2DLod<MappedType>(texHandle, uv, mipLevel);
    return ConvertType::Convert(t);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv) const
{
    bool isResident = false;
    MappedType t = tex2D<MappedType>(texHandle, uv[0], uv[1], &isResident);
    return (isResident) ? ConvertType::Convert(t) : std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv, Vector2 dpdx,
                                              Vector2 dpdy) const
{
    bool isResident = false;
    MappedType t = tex2DGrad<MappedType>(texHandle, uv[0], uv[1],
                                         {dpdx[0], dpdx[1]},
                                         {dpdy[0], dpdy[1]}, &isResident);
    return (isResident) ? ConvertType::Convert(t) : std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<2, T>::operator()(Vector2 uv, Float mipLevel) const
{
    bool isResident = false;
    MappedType t = tex2DLod<MappedType>(texHandle, uv[0], uv[1], mipLevel,
                                        &isResident);
    return (isResident) ? ConvertType::Convert(t) : std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<3, T>::operator()(Vector3 uv) const
{
    bool isResident = false;
    MappedType t = tex3D<MappedType>(texHandle,
                                     uv[0], uv[1], uv[2],
                                     &isResident);
    return (isResident) ? ConvertType::Convert(t) : std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<3, T>::operator()(Vector3 uv, Vector3 dpdx,
                                              Vector3 dpdy) const
{
    bool isResident = false;
    MappedType t = tex3DGrad<MappedType>(texHandle,
                                         uv[0], uv[1], uv[2],
                                         {dpdx[0], dpdx[1], dpdx[2]},
                                         {dpdy[0], dpdy[1], dpdy[2]},
                                         &isResident);
    return (isResident) ? ConvertType::Convert(t) : std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureViewCUDA<3, T>::operator()(Vector3 uv, Float mipLevel) const
{
    bool isResident = false;
    MappedType t = tex3DLod<MappedType>(texHandle,
                                        uv[0], uv[1], uv[2],
                                        mipLevel,
                                        &isResident);
    return (isResident) ? ConvertType::Convert(t) : std::nullopt;
}

}