#pragma once

namespace mray::cuda
{

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>::RWTextureRefCUDA(cudaSurfaceObject_t sIn)
    : s(sIn)
{}

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>::RWTextureRefCUDA(RWTextureRefCUDA&& other)
    : s(std::exchange(other.s, cudaSurfaceObject_t(0)))
{}

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>& RWTextureRefCUDA<D, T>::operator=(RWTextureRefCUDA&& other)
{
    assert(this != &other);
    CUDA_CHECK(cudaDestroySurfaceObject(s));
    s = std::exchange(other.s, cudaSurfaceObject_t(0));
    return *this;
}

template <uint32_t D, class T>
RWTextureRefCUDA<D, T>::~RWTextureRefCUDA()
{
    CUDA_CHECK(cudaDestroySurfaceObject(s));
}

template <uint32_t D, class T>
RWTextureViewCUDA<D, T> RWTextureRefCUDA<D, T>::View() const
{
    return RWTextureViewCUDA<D, T>(s);
}

template<uint32_t D, class T>
template<class QT>
requires(std::is_same_v<QT, T>)
TextureViewCUDA<D, QT> TextureCUDA_Normal<D, T>::View() const
{
    // Normalize integers requested bu view is created with the same type
    if(texParams.normIntegers)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equivalents) "
                        "for normalized integers");
    return TextureViewCUDA<D, QT>(tex);
}

template<uint32_t D, class T>
template<class QT>
requires(!std::is_same_v<QT, T> &&
         (VectorTypeToChannels<T>() ==
          VectorTypeToChannels<QT>()))
TextureViewCUDA<D, QT> TextureCUDA_Normal<D, T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                 std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(texParams.normCoordinates && !IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equivalents) "
                        "for normalized integers");
    else if(texParams.normCoordinates && IsFloatType)
        return TextureViewCUDA<D, QT>(tex);
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
TextureViewCUDA<2, QT> TextureCUDA_BC<T>::View() const
{
    constexpr bool IsFloatType = (std::is_same_v<QT, Float> ||
                                  std::is_same_v<QT, Vector<ChannelCount, Float>>);
    if(!IsFloatType)
        throw MRayError("Unable to create a view of texture. "
                        "View type must be \"Float\" (or vector equivalents) "
                        "for block compressed textures (BC1-7)");
    return TextureViewCUDA<2, QT>(tex);
};


}
