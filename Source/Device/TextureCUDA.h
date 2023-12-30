#pragma once

#include "TextureViewCUDA.h"
#include "GPUTypes.h"

namespace mray::cuda
{

class GPUDeviceCUDA;

template<int D, class T>
class TextureCUDA
{
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
    static_assert(is_TextureType_v<T>, "Invalid texture type");

    using UnderlyingType = typename TextureInitParams<D, T>::UnderlyingType;
    //using CudaType = ....;
    //static constexpr cudaExtent

    public:

    private:
    const GPUDeviceCUDA&        gpu;
    cudaTextureObject_t         tex;
    cudaMipmappedArray_t        data = nullptr;
    TextureInitParams<D, T>     textParams;
    size_t                      size;

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

    template<int D, class QT>
    TextureViewCUDA<D, QT> View();
};

}