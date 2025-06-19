#pragma once

#include <cstdint>
#include "Core/TypeFinder.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include "Device/GPUTypes.h"

namespace mray::host
{

template<uint32_t DIM, class T>
class TextureViewCPU
{
    public:
    static constexpr uint32_t Channels = VectorTypeToChannels<T>();
    static constexpr uint32_t Dimensions = DIM;

    private:
    using UV            = UVType<DIM>;

    private:

    public:
    MRAY_HOST               TextureViewCPU() {}
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

        private:
        T&                  pixel;
        // Constructor
        MRAY_GPU            PixRef(T& pixel);
        public:
        MRAY_GPU PixRef&    operator=(const T&);
    };
    static constexpr uint32_t Channels = PixRef::Channels;
    using Type              = T;
    using PaddedChannelType = T;

    private:
    Span<T>         pixels;

    public:
    // Full Texture object access
    MRAY_HOST       RWTextureViewCPU() {}
    // Write
    MRAY_GPU PixRef operator()(TextureExtent<DIM>);
    // Read
    MRAY_GPU T      operator()(TextureExtent<DIM>) const;
};

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv) const
{
    return T();
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv, UV dpdx, UV dpdy) const
{
    return T();
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T TextureViewCPU<D, T>::operator()(UV uv, Float mipLevel) const
{
    return T();
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
RWTextureViewCPU<D, T>::PixRef::PixRef(T& p)
    : pixel(p)
{}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewCPU<D, T>::PixRef&
RWTextureViewCPU<D, T>::PixRef::operator=(const T& val)
{
    pixel = val;
    return *this;
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
typename RWTextureViewCPU<D, T>::PixRef
RWTextureViewCPU<D, T>::operator()(TextureExtent<D> ij)
{
    return PixRef(pixels[0]);
}

template<uint32_t D, class T>
MRAY_GPU MRAY_GPU_INLINE
T RWTextureViewCPU<D, T>::operator()(TextureExtent<D> ij) const
{
    return T();
}

}