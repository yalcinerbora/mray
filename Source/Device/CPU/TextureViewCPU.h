#pragma once
// IWYU pragma: private; include "GPUTextureView.h"

#include <cstdint>

#include "Core/Vector.h"
#include "Core/Types.h"
#include "Core/MRayDataType.h"
#include "Core/GraphicsFunctions.h"

#include "Device/GPUTypes.h"

namespace mray::host
{

template<class T>
concept FloatPixelC =
(
    std::is_same_v<T, Float>    ||
    std::is_same_v<T, Vector2>  ||
    std::is_same_v<T, Vector3>  ||
    std::is_same_v<T, Vector4>
    // TODO: Add half here when it is supported
);

template<uint32_t D, class T>
class TextureViewCPU
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = D;

    private:
    using UV            = UVType<D>;
    struct Interpolants
    {
        TextureSignedExtent<D>  startPixel;
        std::array<Float, D>    fractions;
    };

    private:
    // Type erease data and params
    const Byte* const*          dataPtr;
    const TextureInitParams<D>* texParams;
    MRayPixelTypeRT             dt;

    private:
    template<uint32_t C, bool IsSigned, class PixelType>
    MRAY_GPU T              Convert(const PixelType&) const;
    MRAY_GPU T              ReadPixel(TextureExtent<D> ijk,
                                      TextureExtent<D> mipSize,
                                      size_t mipPixelStartOffset) const;
    MRAY_GPU T              ReadInterpolatedPixel(UV uv, TextureExtent<D> mipSize,
                                                  size_t mipPixelStartOffset) const requires(FloatPixelC<T>);
    MRAY_GPU T              ReadInterpolatedPixel(UV uv, TextureExtent<D> mipSize,
                                                  size_t mipPixelStartOffset) const requires(!FloatPixelC<T>);
    MRAY_GPU
    TextureExtent<D>        ResolveEdge(TextureSignedExtent<D> ijk,
                                        TextureExtent<D> mipSize) const;
    MRAY_GPU
    TextureSignedExtent<D>  NearestPixel(UV texel) const;
    MRAY_GPU
    Interpolants            FindInterpolants(UV uv, TextureExtent<D> mipSize) const;
    MRAY_GPU UV             IfUVConvertToTexel(UV texelOrUV, TextureExtent<D> mipSize) const;
    MRAY_GPU UV             IfTexelConvertToUV(UV texelOrUV, TextureExtent<D> mipSize) const;

    public:
    MRAY_HOST               TextureViewCPU(const Byte* const* data,
                                           const TextureInitParams<D>* texParams,
                                           MRayPixelTypeRT dt);
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
class RWTextureViewCPU
{
    public:
    class PixRef
    {
        friend class RWTextureViewCPU<DIM, T>;

        public:
        // Texture channels
        static constexpr uint32_t Channels = VectorTypeToChannels<T>();
        using PaddedChannelType = PaddedChannel<Channels, T>;

        private:
        PaddedChannelType&  pixel;
        // Constructor
        MRAY_GPU            PixRef(PaddedChannelType& pixel);
        public:
        MRAY_GPU PixRef&    operator=(const T&);
    };
    static constexpr uint32_t Channels = PixRef::Channels;
    using Type              = T;
    using PaddedChannelType = typename PixRef::PaddedChannelType;

    private:
    Span<PaddedChannelType> pixels;
    TextureExtent<DIM>      dim;

    public:
    // Full Texture object access
    MRAY_HOST       RWTextureViewCPU(Span<PaddedChannelType> pixels,
                                     TextureExtent<DIM> dim);
    // Write
    MRAY_GPU PixRef operator()(TextureExtent<DIM>);
    // Read
    MRAY_GPU T      operator()(TextureExtent<DIM>) const;
};

// Some Inner Type metaprogramming, CPU compiler
// needs Inner type of types that does not have inner type
// to be valid (compile time short circuiting is not available)
//
// i.e.
// if constexpr(C == 3 && std::is_same_v<InnerType<PixelType>, InnerType<T>>)
// C == 3 means it is 3 channel type then Pixel type is Vector3X and it
// has a inner type. However this generic code is instantiated with
// multitude of types one of which is "C == 1, Float".
// Code must be semantically correct I gusess?
//
// So we give "void" when PixelType is arithmetic type
template<class T> struct InnerTypeT { using Type = void; };
template<VectorC T> struct InnerTypeT<T> { using Type = typename T::InnerType; };
template<class T> using InnerType = typename InnerTypeT<T>::Type;

template<uint32_t D, class T>
template<uint32_t C, bool IsSigned, class PixelType>
MRAY_GPU
T TextureViewCPU<D, T>::Convert(const PixelType& pixel) const
{
    // Fetch the inner type if any (for multi-channel types)
    using namespace Bit::NormConversion;

    // Identity Conversion read type and query type is same
    if constexpr(std::is_same_v<PixelType, T>)
        return pixel;
    // Padded channel to channel
    else if constexpr(C == 3 && std::is_same_v<InnerType<PixelType>, InnerType<T>>)
        return T(pixel);
    // Single Channel UNorm/SNorm conversion
    else if constexpr(C == 1 && IsNormConvertible<PixelType>() &&
                      std::is_same_v<T, Float>)
    {
        if constexpr(IsSigned)  return FromSNorm<T>(pixel);
        else                    return FromUNorm<T>(pixel);
    }
    // Multi Channel UNorm/SNorm conversion
    else if constexpr(C != 1 && IsNormConvertible<PixelType>() &&
                      std::is_same_v<InnerType<T>, Float>)
    {
        T result;
        UNROLL_LOOP
        for(uint32_t i = 0; i < C; i++)
        {
            if constexpr(IsSigned)  result[i] = FromSNorm<Float>(pixel[i]);
            else                    result[i] = FromUNorm<Float>(pixel[i]);
        }
        return result;
    }
    // Query data <=> Read data mismatch
    // just return default T
    else return T();
}

template<uint32_t D, class T>
MRAY_GPU
UVType<D> TextureViewCPU<D, T>::IfUVConvertToTexel(UV texelOrUV,
                                                   TextureExtent<D> mipSize) const
{
    bool isUV = texParams->normCoordinates;
    if(isUV)    return texelOrUV * UV(mipSize);
    else        return texelOrUV;
}

template<uint32_t D, class T>
MRAY_GPU
UVType<D> TextureViewCPU<D, T>::IfTexelConvertToUV(UV texelOrUV,
                                                   TextureExtent<D> mipSize) const
{
    bool isTexel = !texParams->normCoordinates;
    if(isTexel) return texelOrUV / UV(mipSize);
    else        return texelOrUV;
}

template<uint32_t D, class T>
MRAY_GPU
TextureExtent<D> TextureViewCPU<D, T>::ResolveEdge(TextureSignedExtent<D> ijk,
                                                   TextureExtent<D> mipSizeIn) const
{
    auto mipSize = TextureSignedExtent<D>(mipSizeIn);
    using VecXi = TextureSignedExtent<D>;

    switch(texParams->eResolve)
    {
        using enum MRayTextureEdgeResolveEnum;
        case MR_CLAMP:
        {
            if constexpr(D == 1)    ijk = Math::Clamp(ijk, 0, mipSize);
            else                    ijk = ijk.Clamp(VecXi(0), mipSize);
            break;
        }
        case MR_MIRROR:
        {
            VecXi dim = ijk / mipSize;
            ijk = ijk % mipSize;

            if constexpr(D == 1)
            {
                if(ijk < 0)             ijk += mipSize;
                if((dim & 0x1) == 1)    ijk = mipSize - ijk;
            }
            else for(uint32_t i = 0; i < D; i++)
            {
                if(ijk[i] < 0)          ijk[i] += mipSize[i];
                if((dim[i] & 0x1) == 1) ijk[i] = mipSize[i] - ijk[i];
            }
            break;
        }
        case MR_WRAP:
        {
            ijk = ijk % mipSize;
            if constexpr(D == 1)
            {
                if(ijk < 0)     ijk += mipSize;
            }
            else for(uint32_t i = 0; i < D; i++)
            {
                if(ijk[i] < 0)  ijk[i] += mipSize[i];
            }
            break;
        }
        default: break;
    }
    //
    assert(ijk >= VecXi(0));
    return TextureExtent<D>(ijk);
}

template<uint32_t D, class T>
MRAY_GPU
TextureSignedExtent<D> TextureViewCPU<D, T>::NearestPixel(UV texel) const
{
    TextureSignedExtent<D> result;
    if constexpr(D == 1)
        result = int32_t(std::round(texel - Float(0.5)));
    else for(uint32_t i = 0; i < D; i++)
    {
        result[i] = int32_t(std::round(texel[i] - Float(0.5)));
    }
    return TextureSignedExtent<D>(result);
}

template<uint32_t D, class T>
MRAY_GPU
typename TextureViewCPU<D, T>::Interpolants
TextureViewCPU<D, T>::FindInterpolants(UV uv, TextureExtent<D> mipSize) const
{
    Interpolants result = {};
    UV texel = uv * UV(mipSize) - UV(0.5);
    if constexpr(D == 1)
    {
        Float base;
        result.fractions[0] = std::modf(texel, &base);
        result.startPixel = TextureSignedExtent<D>(base);
        // Make modf to behave as if it is positive
        if(result.fractions[0] < Float(0))
        {
            result.startPixel -= 1;
            result.fractions[0] = std::abs(result.fractions[0]);
        }
    }
    else for(uint32_t i = 0; i < D; i++)
    {
        Float base;
        result.fractions[i] = std::modf(texel[i], &base);
        result.startPixel[i] = int32_t(base);
        // Make modf to behave as if it is positive
        if(result.fractions[i] < Float(0))
        {
            result.startPixel[i] -= 1;
            result.fractions[i] = std::abs(result.fractions[i]);
        }
    }
    return result;
}

template<uint32_t D, class T>
MRAY_GPU
T TextureViewCPU<D, T>::ReadPixel(TextureExtent<D> ijk,
                                  TextureExtent<D> mipSize,
                                  size_t mipPixelStartOffset) const
{
    uint32_t linearIndex = 0;
    if constexpr(D == 1) linearIndex = ijk;
    if constexpr(D == 2) linearIndex = (ijk[1] * mipSize[0] +
                                        ijk[0]);
    if constexpr(D == 3) linearIndex = (ijk[2] * mipSize[1] * mipSize[0] +
                                        ijk[1] * mipSize[0] +
                                        ijk[0]);

    T pixResult = std::visit([&](auto&& pt) -> T
    {
        using EnumT = std::remove_cvref_t<decltype(pt)>;
        static constexpr uint32_t OutC = VectorTypeToChannels<T>();
        static constexpr uint32_t C = EnumT::ChannelCount;
        static constexpr uint32_t IsSigned = EnumT::IsSigned;
        static constexpr uint32_t IsBCPixel = EnumT::IsBCPixel;
        using PixelType = typename EnumT::Type;
        // Skip, stuff that do not get compiled
        if constexpr(IsBCPixel)                 return T();
        else if constexpr((OutC != 3 || C != 4) && OutC != C)  return T();
        // Rest should be fine
        else
        {
            const PixelType* pixels = reinterpret_cast<const PixelType*>(*dataPtr);
            pixels += mipPixelStartOffset;
            return Convert<OutC, IsSigned, PixelType>(pixels[linearIndex]);
        }
    }, dt);
    return pixResult;
}

template<uint32_t D, class T>
MRAY_GPU
T TextureViewCPU<D, T>::ReadInterpolatedPixel(UV, TextureExtent<D>, size_t) const
requires (!FloatPixelC<T>)
{
    return T();
}

template<uint32_t D, class T>
MRAY_GPU
T TextureViewCPU<D, T>::ReadInterpolatedPixel(UV uv, TextureExtent<D> mipSize,
                                              size_t mipPixelStartOffset) const
requires (FloatPixelC<T>)
{
    auto [ijkStart, fractions] = FindInterpolants(uv, mipSize);
    // Fill a local buffer for interp
    static constexpr uint32_t DATA_PER_LERP = (1u << (D));
    std::array<T, DATA_PER_LERP> pix;
    //
    static constexpr uint32_t K = (D >= 3) ? 2u : 1u;
    static constexpr uint32_t J = (D >= 2) ? 2u : 1u;
    static constexpr uint32_t I = (D >= 1) ? 2u : 1u;
    for(uint32_t k = 0; k < K; k++)
    for(uint32_t j = 0; j < J; j++)
    for(uint32_t i = 0; i < I; i++)
    {
        TextureSignedExtent<D> ijk = ijkStart;
        if constexpr(D == 1)        ijk += i;
        else if constexpr(D == 2)   ijk += TextureSignedExtent<D>(i, j);
        else                        ijk += TextureSignedExtent<D>(i, j, k);
        //
        uint32_t localIndex = (k << 2) + (j << 1) + i;
        pix[localIndex] = ReadPixel(ResolveEdge(ijk, mipSize), mipSize,
                                    mipPixelStartOffset);
    }
    // Lerp part
    for(uint32_t pass = D; pass > 0; pass--)
    for(uint32_t i = 0; i < pass; i++)
        if constexpr(std::is_same_v<Float, T>)
            pix[i] = Math::Lerp(pix[i * 2], pix[i * 2 + 1], fractions[i]);
        else
            pix[i] = T::Lerp(pix[i * 2], pix[i * 2 + 1], fractions[i]);
    //
    return pix[0];
}

template<uint32_t D, class T>
MRAY_HOST
TextureViewCPU<D, T>::TextureViewCPU(const Byte* const* data,
                                     const TextureInitParams<D>* tp,
                                     MRayPixelTypeRT dt)
    : dataPtr(data)
    , texParams(tp)
    , dt(dt)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv) const
{
    if(texParams->interp != MRayTextureInterpEnum::MR_LINEAR)
    {
        UV texel = IfUVConvertToTexel(uv, texParams->size);
        TextureSignedExtent<D> texelInt = NearestPixel(texel);
        return ReadPixel(ResolveEdge(texelInt, texParams->size),
                         texParams->size, 0u);
    }
    else
    {
        uv = IfTexelConvertToUV(uv, texParams->size);
        return ReadInterpolatedPixel(uv, texParams->size, 0u);
    }
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv, UV dpdx, UV dpdy) const
{
    // TODO: We ignore AF, change this later
    if(!texParams->normCoordinates)
    {
        dpdx *= UV(texParams->size);
        dpdy *= UV(texParams->size);
    }
    Float maxLenSqr = std::max(dpdx.LengthSqr(), dpdy.LengthSqr());
    Float mipLevel = Float(0.5) * std::log2(maxLenSqr);

    return (*this)(uv, mipLevel);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv, Float mipLevel) const
{
    using Graphics::TextureMipPixelStart;
    using Graphics::TextureMipSize;

    // Mip Level can be [-inf, +inf]
    // -inf or very large negative value can happen
    // when differentials are small
    // Other one is probably never happen but we support it
    mipLevel = Math::Clamp(mipLevel, Float(0), Float(texParams->mipCount - 1));
    Float m0F;
    Float frac = std::modf(mipLevel, &m0F);
    uint32_t m0 = uint32_t(m0F);
    uint32_t m1 = Math::Clamp(m0 + 1u, 0u, texParams->mipCount - 1);

    if(texParams->interp != MRayTextureInterpEnum::MR_LINEAR)
    {
        uint32_t nearestMipLevel = (frac < Float(0.5)) ? m0 : m1;
        uint32_t mipOffset = TextureMipPixelStart(texParams->size,
                                                  nearestMipLevel);
        TextureExtent<D> mipSize = TextureMipSize(texParams->size,
                                                  nearestMipLevel);
        UV texel = IfUVConvertToTexel(uv, mipSize);
        TextureSignedExtent<D> texelInt = NearestPixel(texel);
        return ReadPixel(ResolveEdge(texelInt, texParams->size),
                         mipSize, mipOffset);
    }
    else
    {
        // TODO: Check GPU APIs (CUDA etc.)
        // how they implement "textureLOD" functions
        // when texture is texel accessible
        // given texel coordinates must be on LOD level
        // or base level?
        // Below we assume it is base level
        UV readUV = IfTexelConvertToUV(uv, texParams->size);

        auto FetchPix = [&, this](uint32_t mip) -> T
        {
            uint32_t mipOffset = TextureMipPixelStart(texParams->size, mip);
            TextureExtent<D> mipSize = TextureMipSize(texParams->size, mip);
            return ReadInterpolatedPixel(readUV, mipSize, mipOffset);
        };

        T pix0  = FetchPix(m0);
        if(m0 == m1) return pix0;
        T pix1 = FetchPix(m1);

        if constexpr(std::is_same_v<Float, T>)
            return Math::Lerp(pix0, pix1, frac);
        else
            return T::Lerp(pix0, pix1, frac);
    }
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
RWTextureViewCPU<D, T>::PixRef::PixRef(PaddedChannelType& p)
    : pixel(p)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewCPU<D, T>::PixRef&
RWTextureViewCPU<D, T>::PixRef::operator=(const T& val)
{
    if constexpr(std::is_same_v<PaddedChannelType, T>)
        pixel = val;
    else
        pixel = PaddedChannelType(val, 0);

    return *this;
}

template<uint32_t D, class T>
MRAY_HOST inline
RWTextureViewCPU<D, T>::RWTextureViewCPU(Span<PaddedChannelType> pixels,
                                         TextureExtent<D> dim)
    : pixels(pixels)
    , dim(dim)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewCPU<D, T>::PixRef
RWTextureViewCPU<D, T>::operator()(TextureExtent<D> ij)
{
    if constexpr(D == 1)        return PixRef(pixels[ij]);
    else if constexpr(D == 2)   return PixRef(pixels[ij[1] * dim[0] + ij[0]]);
    else                        return PixRef(pixels[ij[1] * dim[0] * dim[1] +
                                              ij[1] * dim[0] + ij[0]]);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T RWTextureViewCPU<D, T>::operator()(TextureExtent<D> ij) const
{
    if constexpr(D == 1)        return T(pixels[ij]);
    else if constexpr(D == 2)   return T(pixels[ij[1] * dim[0] + ij[0]]);
    else                        return T(pixels[ij[1] * dim[0] * dim[1] +
                                                ij[1] * dim[0] + ij[0]]);
}

}