#pragma once

#include "TextureCPU.h"

namespace mray::host
{

template <uint32_t D, class T>
RWTextureRefCPU<D, T>::RWTextureRefCPU(PaddedChannelType* const* data,
                                       size_t mipStartOffset,
                                       TextureExtent<D> dim)
    : data(data)
    , mipStartOffset(mipStartOffset)
    , dim(dim)
{}

template <uint32_t D, class T>
RWTextureViewCPU<D, T> RWTextureRefCPU<D, T>::View() const
{
    size_t mipPixelCount = 0;
    if constexpr(D == 1)
        mipPixelCount = dim;
    else
        mipPixelCount = dim.Multiply();
    Span<PaddedChannelType> dataSpan(*data + mipStartOffset,
                                     mipPixelCount);
    return RWTextureViewCPU<D, T>(dataSpan, dim);
}

template<uint32_t D, class T>
template<class QT>
requires(std::is_same_v<QT, T>)
TextureViewCPU<D, QT> TextureCPU_Normal<D, T>::View() const
{
    // Normalize integers requested bu view is created with the same type
    if(texParams.normIntegers)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equivalents) "
                        "for normalized integers");
    const Byte* const* byteData = reinterpret_cast<const Byte* const*>(&dataPtr);
    static constexpr auto PixelEnum = static_cast<MRayPixelEnum>(PixelTypeToEnum::template Find<PaddedChannelType>);
    return TextureViewCPU<D, QT>(byteData, &texParams,
                                 MRayPixelTypeRT(MRayPixelType<PixelEnum>()));
}

template<uint32_t D, class T>
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (PixelTypeToChannels<T>() == PixelTypeToChannels<QT>()))
TextureViewCPU<D, QT> TextureCPU_Normal<D, T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                 std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(texParams.normCoordinates && !IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equivalents) "
                        "for normalized integers");
    else if(texParams.normCoordinates && IsFloatType)
    {
        const Byte* const* byteData = reinterpret_cast<const Byte* const*>(&dataPtr);
        static constexpr auto PixelEnum = static_cast<MRayPixelEnum>(PixelTypeToEnum::template Find<PaddedChannelType>);
        return TextureViewCPU<D, QT>(byteData, &texParams,
                                     MRayPixelTypeRT(MRayPixelType<PixelEnum>()));
    }
    else
    {
        // Now we have integer types and these will not be fetched
        // as normalized floats. This means type conversion either narrowing
        // or expanding. (i.e., "short -> int" or "int -> short").
        //
        // Do not support these currently, I dunno if it compiles down to
        // auto narrowing or texture fetch functions will fail.
        //
        // TODO: change this later if all is OK
        throw MRayError("Unable to create a view of texture. "
                        "Any type conversion (narrowing or expanding) is not supported on textures."
                        " This function should only be called for normalized integers with \"Float\" types");
    }
}

template<class T>
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (BCTypeToChannels<T>() == VectorTypeToChannels<QT>()))
TextureViewCPU<2, QT> TextureCPU_BC<T>::View() const
{
    throw MRayError("CPU Device does not support BC textures!");
};

}
