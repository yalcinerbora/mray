#pragma once

#include "GPUTypes.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/TextureViewCUDA.h"

    // Alias the types
    template<uint32_t DIM, class T>
    using RWTextureView = mray::cuda::RWTextureViewCUDA<DIM, T>;

    template<uint32_t DIM, class T>
    using TextureView = mray::cuda::TextureViewCUDA<DIM, T>;

#elif defined MRAY_GPU_BACKEND_HIP
    #include "HIP/TextureViewHIP.h"

    // Alias the types
    template<uint32_t DIM, class T>
    using RWTextureView = mray::hip::RWTextureViewHIP<DIM, T>;

    template<uint32_t DIM, class T>
    using TextureView = mray::hip::TextureViewHIP<DIM, T>;

#elif defined GPU_BACKEND_HOST
    #error Not yet!
#else
    #error Please define a GPU Backend!
#endif