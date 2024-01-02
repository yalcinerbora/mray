#pragma once

#include "DeviceMemoryCUDA.h"
#include "TextureViewCUDA.h"
#include "DefinitionsCUDA.h"
#include "GPUSystemCUDA.h"
#include "GPUTypes.h"

namespace mray::cuda
{

template <class T>
constexpr bool IsNormConvertibleCUDA()
{
    // YOLO
    return (std::is_same_v<T, uint32_t>     ||
            std::is_same_v<T, Vector2ui>    ||
            std::is_same_v<T, Vector3ui>    ||
            std::is_same_v<T, Vector4ui>    ||

            std::is_same_v<T, int32_t>      ||
            std::is_same_v<T, Vector2i>     ||
            std::is_same_v<T, Vector3i>     ||
            std::is_same_v<T, Vector4i>     ||

            std::is_same_v<T, uint16_t>     ||
            std::is_same_v<T, Vector2us>    ||
            std::is_same_v<T, Vector3us>    ||
            std::is_same_v<T, Vector4us>    ||

            std::is_same_v<T, int16_t>      ||
            std::is_same_v<T, Vector2s>     ||
            std::is_same_v<T, Vector3s>     ||
            std::is_same_v<T, Vector4s>     ||

            std::is_same_v<T, uint8_t>      ||
            std::is_same_v<T, Vector2uc>    ||
            std::is_same_v<T, Vector3uc>    ||
            std::is_same_v<T, Vector4uc>    ||

            std::is_same_v<T, int8_t>       ||
            std::is_same_v<T, Vector2c>     ||
            std::is_same_v<T, Vector3c>     ||
            std::is_same_v<T, Vector4c>);
}

template<int D, class T>
class TextureCUDA
{
    using UnderlyingType = typename TextureInitParams<D, T>::UnderlyingType;
    using CudaType = typename decltype(VectorTypeToCUDA<T>())::MappedType;
    static constexpr uint32_t ChannelCount = VectorTypeToChannels<T>().Channels;
    static constexpr bool IsNormConvertible = IsNormConvertibleCUDA<T>();

    // Sanity Checks
    static_assert(std::is_same_v<UnderlyingType, T>);
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");

    public:

    private:
    const GPUDeviceCUDA*        gpu;
    cudaTextureObject_t         tex         = (cudaTextureObject_t)0;
    cudaMipmappedArray_t        data        = nullptr;
    TextureInitParams<D, T>     texParams;

    // Allocation related
    bool                        allocated   = false;
    size_t                      size        = 0;
    size_t                      alignment   = 0;

    protected:
    public:
    // Constructors & Destructor
                            TextureCUDA() = delete;
                            TextureCUDA(const GPUDeviceCUDA& device,
                                        const TextureInitParams<D, T>& p);
                            TextureCUDA(const TextureCUDA&) = delete;
                            TextureCUDA(TextureCUDA&&) noexcept;
    TextureCUDA&            operator=(const TextureCUDA&) = delete;
    TextureCUDA&            operator=(TextureCUDA&&) noexcept;
                            ~TextureCUDA();

    // Direct view conversion (simple case)
    template<class QT>
    requires(std::is_same_v<QT, T>)
    TextureViewCUDA<D, QT>  View() const;

    template<class QT>
    requires(VectorTypeToChannels<T>().Channels ==
             VectorTypeToChannels<QT>().Channels)
    TextureViewCUDA<D, QT>  View() const;

    size_t                  Size() const;
    size_t                  Alignment() const;
    void                    CommitMemory(const GPUQueueCUDA& queue,
                                         const DeviceLocalMemoryCUDA& deviceMem,
                                         size_t offset);
};

}

#include "TextureCUDA.hpp"

// Common Textures 2D
extern template class mray::cuda::TextureCUDA<2, Float>;
extern template class mray::cuda::TextureCUDA<2, Vector2>;
extern template class mray::cuda::TextureCUDA<2, Vector3>;
extern template class mray::cuda::TextureCUDA<2, Vector4>;

extern template class mray::cuda::TextureCUDA<2, uint8_t>;
extern template class mray::cuda::TextureCUDA<2, Vector2uc>;
extern template class mray::cuda::TextureCUDA<2, Vector3uc>;
extern template class mray::cuda::TextureCUDA<2, Vector4uc>;

extern template class mray::cuda::TextureCUDA<2, int8_t>;
extern template class mray::cuda::TextureCUDA<2, Vector2c>;
extern template class mray::cuda::TextureCUDA<2, Vector3c>;
extern template class mray::cuda::TextureCUDA<2, Vector4c>;

extern template class mray::cuda::TextureCUDA<2, uint16_t>;
extern template class mray::cuda::TextureCUDA<2, Vector2us>;
extern template class mray::cuda::TextureCUDA<2, Vector3us>;
extern template class mray::cuda::TextureCUDA<2, Vector4us>;

extern template class mray::cuda::TextureCUDA<2, int16_t>;
extern template class mray::cuda::TextureCUDA<2, Vector2s>;
extern template class mray::cuda::TextureCUDA<2, Vector3s>;
extern template class mray::cuda::TextureCUDA<2, Vector4s>;