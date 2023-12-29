#pragma once

//#include <cuda>

#include <cstdint>
#include "TextureC.h"
#include "TypeFinder.h"
#include "Vector.h"
#include "Types.h"

typedef uint64_t cudaTextureObject_t;

struct float2{};
struct float4{};
struct int2{};
struct int4{};
struct short2{};
struct short4{};
struct char2{};
struct char4{};

namespace TextureTypesCUDA
{
    //
    template <class Query, class Mapped>
    struct Mapping
    {
        using QueryType = Query;
        using MappedType = Mapped;
    };

    template<class T>
    static constexpr auto VectorTypeToCUDA()
    {
        constexpr auto CUDAToTexTypeMappingList = std::make_tuple
        (
            // 1 Channel
            Mapping<uint32_t, uint32_t>,
            Mapping<float, float>,
            Mapping<double, double>,
            Mapping<Vector2f, float2>,
            Mapping<Vector3f, float4>,
            Mapping<Vector4f, float4>
        );
        return TypeFinder::GetFunctionTuple<T>(std::forward(CUDAToTexTypeMappingList));
    }
}

template<uint32_t DIM, class T>
class TextureCUDA;

// Specializiation of 2D for CUDA
template<class T>
class TextureCUDA<2, T>
{
    private:
    //using MappedType = TextureTypesCUDA::VectorTypeToCUDA<T>()::MappedType;
    cudaTextureObject_t tex;

    public:
    // Base Access
    MRAY_HYBRID Optional<T> operator()(Vector2 uv) const;
    // Gradient Access
    MRAY_HYBRID Optional<T> operator()(Vector2 uv, Vector2 dpdx,
                                       Vector2 dpdy) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<T> operator()(Vector2 uv, uint32_t mipLevel) const;
};
//static_assert(TextureConceptC<TextureCUDA<2, Float>, 2, Float>);
//static_assert(TextureConceptC<TextureCUDA<2, Vector2>, 2, Vector2>);
//static_assert(TextureConceptC<TextureCUDA<2, Vector3>, 2, Vector3>);
//static_assert(TextureConceptC<TextureCUDA<2, int32_t>, 2, int32_t>);
//static_assert(TextureConceptC<TextureCUDA<2, Vector2i>, 2, Vector2i>);
//static_assert(TextureConceptC<TextureCUDA<2, Vector3i>, 2, Vector3i>);
//static_assert(TextureConceptC<TextureCUDA<2, uint32_t>, 2, uint32_t>);
//static_assert(TextureConceptC<TextureCUDA<2, Vector2ui>, 2, Vector2ui>);
//static_assert(TextureConceptC<TextureCUDA<2, Vector3ui>, 2, Vector3ui>);

//template<uint32_t DIM, class T>
//using TextureType = TextureCUDA<DIM, T>;

class TextureAccessorCUDA
{
    public:

    template <uint32_t DIM, class T>
    using TextureType = TextureCUDA<DIM, T>;


    //// Specializiation of 2D for CUDA
    //template<>
    //class Texture<2, Spectrum>
    //{
    //    private:
    //    cudaTextureObject_t tex;
    //    cudaTextureObject_t spectrumLookupTable;

    //    public:
    //    // Base Access
    //    MRAY_HYBRID Optional<Spectrum> operator()(Vector2 uv) const;
    //    // Gradient Access
    //    MRAY_HYBRID Optional<Spectrum> operator()(Vector2 uv, Vector2 dpdx,
    //                                              Vector2 dpdy) const;
    //    // Direct Mip Access
    //    MRAY_HYBRID Optional<Spectrum> operator()(Vector2 uv, uint32_t mipLevel) const;
    //};
    //static_assert(TextureConceptC<Texture<2, Spectrum>, 2, Spectrum>);

    //template<uint32_t DIM, class T>
    //class Texture
    //{

    //};
};

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureCUDA<2, T>::operator()(Vector2 uv) const
{
    return T{};
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureCUDA<2, T>::operator()(Vector2 uv, Vector2 dpdx,
                                                      Vector2 dpdy) const
{
    return T{};
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> TextureCUDA<2, T>::operator()(Vector2 uv, uint32_t mipLevel) const
{
    return T{};
}
