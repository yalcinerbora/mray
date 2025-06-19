#pragma once

#include "GPUTypes.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/TextureCUDA.h"

    using TextureBackingMemory = mray::cuda::TextureBackingMemoryCUDA;

    template<uint32_t DIM, class T>
    using Texture = mray::cuda::TextureCUDA<DIM, T>;

    template<uint32_t DIM, class T>
    using RWTextureRef = mray::cuda::RWTextureRefCUDA<DIM, T>;

#elif defined MRAY_GPU_BACKEND_HIP
    #include "HIP/TextureHIP.h"

    using TextureBackingMemory = mray::hip::TextureBackingMemoryHIP;

    template<uint32_t DIM, class T>
    using Texture = mray::hip::TextureHIP<DIM, T>;

    template<uint32_t DIM, class T>
    using RWTextureRef = mray::hip::RWTextureRefHIP<DIM, T>;

#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/TextureCPU.h"

    using TextureBackingMemory = mray::host::TextureBackingMemoryCPU;

    template<uint32_t DIM, class T>
    using Texture = mray::host::TextureCPU<DIM, T>;

    template<uint32_t DIM, class T>
    using RWTextureRef = mray::host::RWTextureRefCPU<DIM, T>;

#else
    #error Please define a GPU Backend!
#endif