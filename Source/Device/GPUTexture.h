#pragma once

#include "GPUTypes.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/TextureCUDA.h"

    // Alias the types
    using TextureBackingMemory = mray::cuda::TextureBackingMemoryCUDA;

    template<uint32_t DIM, class T>
    using Texture = mray::cuda::TextureCUDA<DIM, T>;

    template<uint32_t DIM, class T>
    using RWTextureRef = mray::cuda::RWTextureRefCUDA<DIM, T>;

#elif defined GPU_BACKEND_HOST
    #error Not yet!
#else
    #error Please define a GPU Backend!
#endif