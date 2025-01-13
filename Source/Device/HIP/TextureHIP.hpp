#pragma once

namespace mray::hip
{

template <uint32_t D, class T>
RWTextureRefHIP<D, T>::RWTextureRefHIP(hipSurfaceObject_t sIn)
    : s(sIn)
{}

template <uint32_t D, class T>
RWTextureRefHIP<D, T>::RWTextureRefHIP(RWTextureRefHIP&& other)
    : s(std::exchange(other.s, hipSurfaceObject_t(0)))
{}

template <uint32_t D, class T>
RWTextureRefHIP<D, T>& RWTextureRefHIP<D, T>::operator=(RWTextureRefHIP&& other)
{
    assert(this != &other);
    HIP_CHECK(hipDestroySurfaceObject(s));
    s = std::exchange(other.s, hipSurfaceObject_t(0));
    return *this;
}

template <uint32_t D, class T>
RWTextureRefHIP<D, T>::~RWTextureRefHIP()
{
    HIP_CHECK(hipDestroySurfaceObject(s));
}

template <uint32_t D, class T>
RWTextureViewHIP<D, T> RWTextureRefHIP<D, T>::View() const
{
    return RWTextureViewHIP<D, T>(s);
}

template<uint32_t D, class T>
template<class QT>
requires(std::is_same_v<QT, T>)
TextureViewHIP<D, QT> TextureHIP_Normal<D, T>::View() const
{
    // Normalize integers requested bu view is created with the same type
    if(texParams.normIntegers)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for normalized integers");
    return TextureViewHIP<D, QT>(tex);
}

template<uint32_t D, class T>
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (VectorTypeToChannels<T>() ==
          VectorTypeToChannels<QT>()))
TextureViewHIP<D, QT> TextureHIP_Normal<D, T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                 std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(texParams.normCoordinates && !IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for normalized integers");
    else if(texParams.normCoordinates && IsFloatType)
        return TextureViewHIP<D, QT>(tex);
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
TextureViewHIP<2, QT> TextureHIP_BC<T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                  std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(!IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for block compressed textures (BC1-7)");
    return TextureViewHIP<2, QT>(tex);
};


}
