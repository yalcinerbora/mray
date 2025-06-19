#pragma once

namespace mray::host
{

template <uint32_t D, class T>
RWTextureRefCPU<D, T>::RWTextureRefCPU()
{}

template <uint32_t D, class T>
RWTextureRefCPU<D, T>::RWTextureRefCPU(RWTextureRefCPU&& other)
{}

template <uint32_t D, class T>
RWTextureRefCPU<D, T>& RWTextureRefCPU<D, T>::operator=(RWTextureRefCPU&& other)
{
    assert(this != &other);
    return *this;
}

template <uint32_t D, class T>
RWTextureRefCPU<D, T>::~RWTextureRefCPU()
{
}

template <uint32_t D, class T>
RWTextureViewCPU<D, T> RWTextureRefCPU<D, T>::View() const
{
    return RWTextureViewCPU<D, T>();
}

template<uint32_t D, class T>
template<class QT>
requires(std::is_same_v<QT, T>)
TextureViewCPU<D, QT> TextureCPU_Normal<D, T>::View() const
{
    // Normalize integers requested bu view is created with the same type
    if(texParams.normIntegers)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for normalized integers");
    return TextureViewCPU<D, QT>();
}

template<uint32_t D, class T>
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (VectorTypeToChannels<T>() ==
          VectorTypeToChannels<QT>()))
TextureViewCPU<D, QT> TextureCPU_Normal<D, T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                 std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(texParams.normCoordinates && !IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for normalized integers");
    else if(texParams.normCoordinates && IsFloatType)
        return TextureViewCPU<D, QT>();
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
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                  std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(!IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equavilents) "
                        "for block compressed textures (BC1-7)");
    return TextureViewCPU<2, QT>();
};


}
