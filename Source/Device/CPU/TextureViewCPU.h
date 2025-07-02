#pragma once

#include <cstdint>
#include "Core/TypeFinder.h"
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

template<uint32_t DIM, class T>
class TextureViewCPU
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = DIM;

    private:
    using UV            = UVType<DIM>;

    private:
    // Type erease data and params
    Span<const Byte>                data;
    const TextureInitParams<DIM>*   texParams;
    MRayPixelTypeRT                 dt;

    private:
    template<uint32_t C, bool IsSigned, class PixelType>
    MRAY_GPU T              Convert(const PixelType&) const;
    MRAY_GPU T              ReadPixel(TextureExtent<DIM> ijk,
                                      TextureExtent<DIM> mipSize,
                                      Span<const Byte> mipData) const;

    MRAY_GPU T              ReadInterpolatedPixel(UV uv, TextureExtent<DIM> mipSize,
                                                  Span<const Byte> mipData) const requires(FloatPixelC<T>);
    MRAY_GPU T              ReadInterpolatedPixel(UV uv, TextureExtent<DIM> mipSize,
                                                  Span<const Byte> mipData) const requires(!FloatPixelC<T>);

    public:
    MRAY_HOST               TextureViewCPU(Span<const Byte> data,
                                           const TextureInitParams<DIM>* texParams,
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

template<class T>
struct InnerType
{
    using Type = void;
};

template<VectorC T>
struct InnerType<T>
{
    using Type = typename T::InnerType;
};

template<class T>
using InnerTypeT = typename InnerType<T>::Type;

template<uint32_t D, class T>
template<uint32_t C, bool IsSigned, class PixelType>
MRAY_GPU
T TextureViewCPU<D, T>::Convert(const PixelType& pixel) const
{
    // Fetch the inner type if any (for multi-channel types)
    //using InnerType = std::conditional_t<VectorC<T>, typename T::InnerType, void>;
    using namespace Bit::NormConversion;

    // Identity Conversion read type and query type is same
    if constexpr(std::is_same_v<PixelType, T>)
        return pixel;
    // Padded channel to channel
    else if constexpr(C == 3 && std::is_same_v<InnerTypeT<PixelType>, InnerTypeT<T>>)
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
                      std::is_same_v<InnerTypeT<T>, Float>)
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
T TextureViewCPU<D, T>::ReadPixel(TextureExtent<D> ijk,
                                  TextureExtent<D> mipSize,
                                  Span<const Byte> mipData) const
{
    using VecXui = TextureExtent<D>;
    switch(texParams->eResolve)
    {
        using enum MRayTextureEdgeResolveEnum;
        case MR_CLAMP:
        {
            if constexpr(D == 1)
                ijk = Math::Clamp(ijk, VecXui(0), mipSize);
            else
                ijk = ijk.Clamp(VecXui(0), mipSize);
            break;
        }
        case MR_MIRROR:
        {
            VecXui dim = ijk / mipSize;
            ijk = ijk % mipSize;

            if constexpr(D == 1)
            {
                if((dim & 0x1) == 1) ijk = mipSize - ijk;
            }
            else
            {
                for(uint32_t i = 0; i < D; i++)
                {
                    if((dim[i] & 0x1) == 1) ijk[i] = mipSize[i] - ijk[i];
                }
            }
            break;
        }
        case MR_WRAP:   ijk = ijk % mipSize; break;
        default:        break;
    }
    //
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
        //else if constexpr(OutC != 3 || C != 4 || OutC != C)  return T();
        else if constexpr((OutC != 3 || C != 4) && OutC != C)  return T();
        // Rest should be fine
        else
        {
            const PixelType* pixels = reinterpret_cast<const PixelType*>(mipData.data());
            return Convert<OutC, IsSigned, PixelType>(pixels[linearIndex]);
        }
    }, dt);
    return pixResult;
}


template<uint32_t D, class T>

MRAY_GPU
T TextureViewCPU<D, T>::ReadInterpolatedPixel(UV, TextureExtent<D>,
                                              Span<const Byte>) const
requires (!FloatPixelC<T>)
{
    return T();
}

template<uint32_t D, class T>
MRAY_GPU
T TextureViewCPU<D, T>::ReadInterpolatedPixel(UV uv, TextureExtent<D> mipSize,
                                              Span<const Byte> mipData) const
requires (FloatPixelC<T>)
{
    UV texel = uv * UV(mipSize) - UV(0.5);
    //
    std::array<Float, D> texelFrac, texelBase;
    TextureExtent<D> ijkStart;
    if constexpr(D != 1)
    {
        for(uint32_t i = 0; i < D; i++)
        {
            texelFrac[i] = std::modf(texel[i], &texelBase[i]);
            ijkStart[i] = uint32_t(texelBase[i]);
        }
    }
    else
    {
        texelFrac[0] = std::modf(texel, &texelBase[0]);
        ijkStart = TextureExtent<D>(texelBase[0]);
    }

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
        TextureExtent<D> ijk = ijkStart;
        if constexpr(D == 1)        ijk += i;
        else if constexpr(D == 2)   ijk += TextureExtent<D>(i, j);
        else                        ijk += TextureExtent<D>(i, j, k);
        //
        pix[(k << 2) + (j << 1) + i] = ReadPixel(ijk, mipSize, mipData);
    }
    // Lerp part
    for(uint32_t pass = D; pass > 0; pass--)
    for(uint32_t i = 0; i < pass; i++)
        if constexpr(std::is_same_v<Float, T>)
            pix[i] = Math::Lerp(pix[i * 2], pix[i * 2 + 1], texelFrac[i]);
        else
            pix[i] = T::Lerp(pix[i * 2], pix[i * 2 + 1], texelFrac[i]);
    //
    return pix[0];
}

template<uint32_t D, class T>
MRAY_HOST
TextureViewCPU<D, T>::TextureViewCPU(Span<const Byte> data,
                                     const TextureInitParams<D>* tp,
                                     MRayPixelTypeRT dt)
    : data(data)
    , texParams(tp)
    , dt(dt)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv) const
{
    if(texParams->interp != MRayTextureInterpEnum::MR_LINEAR)
    {
        UV texel = (texParams->normCoordinates)
                    ? uv * UV(texParams->size)
                    : uv;
        TextureExtent<D> ijk = TextureExtent<D>(texel);
        return ReadPixel(ijk, texParams->size, data);
    }
    else
    {
        if(texParams->normCoordinates)
            uv = (uv + UV(0.5)) / UV(texParams->size);
        return ReadInterpolatedPixel(uv, texParams->size, data);
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
    if(texParams->interp != MRayTextureInterpEnum::MR_LINEAR)
    {
        uint32_t mipLevelInt = uint32_t(mipLevel);
        uint32_t offset = Graphics::TextureMipPixelStart(texParams->size,
                                                         mipLevelInt);
        TextureExtent<D> size = Graphics::TextureMipSize(texParams->size, mipLevelInt);
        auto mipData = data.subspan(offset);
        // TODO: Check how textureLOD works when coordinates
        // are not UV (i.e. not [0, 1])
        // given coordinates should be in mip sizes or base mip size???
        UV texel = (texParams->normCoordinates)
                    ? uv * UV(texParams->size)
                    : uv;
        TextureExtent<D> ijk = TextureExtent<D>(texel);
        return ReadPixel(ijk, size, mipData);
    }
    else
    {
        auto FetchPix = [&, this](uint32_t mip) -> T
        {
            using namespace Graphics;
            uint32_t offset = TextureMipPixelStart(texParams->size, mip);
            TextureExtent<D> size = Graphics::TextureMipSize(texParams->size, mip);
            auto mipData = data.subspan(offset);
            UV readUV = (texParams->normCoordinates)
                        ? ((uv + UV(0.5)) / UV(texParams->size))
                        : uv;
            return ReadInterpolatedPixel(readUV, size, mipData);
        };

        Float m0F;
        Float frac = std::modf(mipLevel, &m0F);
        int32_t m0 = int32_t(m0F);
        int32_t m1 = int32_t(mipLevel + Float(0.5));
        m0 = Math::Clamp(m1, 0, int32_t(texParams->mipCount));
        m1 = Math::Clamp(m0, 0, int32_t(texParams->mipCount));

        T pix0  = FetchPix(uint32_t(m0));
        if(m0 == m1) return pix0;
        T pix1 = FetchPix(uint32_t(m1));

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